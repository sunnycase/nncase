// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.Shapes;
using Nncase.Schedule;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Canonicalizes PyNTT packed QKV projection weights to one fixed-capacity,
/// block-local target layout before block microkernel selection.
/// </summary>
public sealed class CanonicalizePackedQKVWeightsPass : ModulePass
{
    private readonly string _moduleKind;

    public CanonicalizePackedQKVWeightsPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var roots = TIRCallGraphRootUtility.Collect(input, _moduleKind);
        if (roots.Length == 0)
        {
            throw new InvalidOperationException(
                $"{nameof(CanonicalizePackedQKVWeightsPass)} found no executable {_moduleKind} TIR root.");
        }

        var canonicalizer = new ModuleCanonicalizer(input, _moduleKind);
        return Task.FromResult(canonicalizer.Run(input.Entry!, roots));
    }

    private sealed class ModuleCanonicalizer
    {
        private readonly IRModule _module;
        private readonly string _moduleKind;
        private readonly Dictionary<PrimFunction, FunctionPlan> _plans = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<PrimFunction> _active = new(ReferenceEqualityComparer.Instance);
        private readonly List<PrimFunction> _postOrder = new();
        private readonly Dictionary<PrimFunction, PrimFunction> _replacements = new(ReferenceEqualityComparer.Instance);

        public ModuleCanonicalizer(IRModule module, string moduleKind)
        {
            _module = module;
            _moduleKind = moduleKind;
        }

        public IRModule Run(BaseFunction entry, IReadOnlyList<PrimFunction> roots)
        {
            foreach (var root in roots)
            {
                Discover(root);
                if (_plans[root].ParameterGroups.Count != 0)
                {
                    throw new InvalidOperationException(
                        $"PyNTT root {root.Name} ABI cannot expose unfused Q/K/V weight parameters. " +
                        "Packed QKV weights must resolve to compiler-owned constants before TIR microkernel selection.");
                }
            }

            foreach (var function in _postOrder)
            {
                var cloner = new FunctionCloner(
                    function,
                    _plans[function],
                    _plans,
                    _replacements);
                var replacement = cloner.CloneFunction();
                _replacements.Add(function, replacement);
            }

            var rootReplacements = roots.ToDictionary(
                root => root,
                root => _replacements[root],
                new ReferenceEqualityComparer<PrimFunction>());
            var wrapperReplacements = TIRCallGraphRootUtility.RebindWrappers(_module, rootReplacements);
            var result = new IRModule();
            foreach (var function in _module.Functions)
            {
                result.Add(function switch
                {
                    PrimFunction prim when _replacements.TryGetValue(prim, out var replacement) => replacement,
                    PrimFunctionWrapper wrapper when wrapperReplacements.TryGetValue(wrapper, out var replacement) => replacement,
                    _ => function,
                });
            }

            result.Entry = entry is PrimFunction primEntry
                ? _replacements[primEntry]
                : entry;
            return result;
        }

        private void Discover(PrimFunction function)
        {
            if (function.ModuleKind != _moduleKind || _plans.ContainsKey(function))
            {
                return;
            }

            if (!_active.Add(function))
            {
                throw new InvalidOperationException(
                    $"Recursive PrimFunction calls are not supported while canonicalizing packed QKV weights: {function.Name}.");
            }

            var calls = CollectCalls(function.Body);
            foreach (var call in calls.Where(call => call.Target is PrimFunction))
            {
                Discover((PrimFunction)call.Target);
            }

            var plan = new FunctionPlan(function);
            foreach (var call in calls.Where(call => call.Target is TIR.NTT.PackedQKVParallelLinear))
            {
                plan.AddQKVCall(call);
            }

            foreach (var call in calls.Where(call => call.Target is PrimFunction))
            {
                var callee = (PrimFunction)call.Target;
                if (_plans.TryGetValue(callee, out var calleePlan))
                {
                    plan.PropagateCalleeGroups(call, calleePlan);
                }
            }

            _active.Remove(function);
            _plans.Add(function, plan);
            _postOrder.Add(function);
        }

        private static Call[] CollectCalls(BaseExpr expression)
        {
            var collector = new CallCollector();
            collector.Visit(expression);
            return collector.Calls.ToArray();
        }
    }

    private sealed class FunctionPlan
    {
        private readonly Dictionary<BufferVar, int> _parameterIndices;
        private readonly Dictionary<ParameterTriple, ParameterGroup> _groups = new();

        public FunctionPlan(PrimFunction function)
        {
            Function = function;
            _parameterIndices = new(ReferenceEqualityComparer.Instance);
            for (var index = 0; index < function.Parameters.Length; index++)
            {
                if (function.Parameters[index] is BufferVar bufferVar)
                {
                    _parameterIndices.Add(bufferVar, index);
                }
            }
        }

        public PrimFunction Function { get; }

        public IReadOnlyList<ParameterGroup> ParameterGroups => _groups.Values.OrderBy(group => group.QIndex).ToArray();

        public Dictionary<Call, FusedWeightLayout> QKVLayouts { get; } = new(ReferenceEqualityComparer.Instance);

        public void AddQKVCall(Call call)
        {
            var op = (TIR.NTT.PackedQKVParallelLinear)call.Target;
            if (op.RhsLayout is not (
                IR.NTT.PackedMatMulRhsLayout.KMajor or
                IR.NTT.PackedMatMulRhsLayout.NMajorKPacked))
            {
                throw new InvalidOperationException(
                    $"PyNTT packed QKV weight canonicalization does not support {op.RhsLayout} weights.");
            }

            var arguments = call.Arguments.ToArray();
            var weights = GetWeightBuffers(arguments, call);
            var outputs = new[]
            {
                RequireBuffer(arguments[13], "Q output", call),
                RequireBuffer(arguments[14], "K output", call),
                RequireBuffer(arguments[15], "V output", call),
            };
            var layout = FusedWeightLayout.Create(
                weights,
                outputs,
                op.RhsLayout,
                op.OutputNVectorLaneCount);
            QKVLayouts.Add(call, layout);

            if (TryGetParameterTriple(weights, out var triple))
            {
                AddGroup(triple, layout, $"kernel call in {Function.Name}");
            }
            else if (!weights.All(IsCanonicalConstantBuffer))
            {
                throw new InvalidOperationException(
                    $"Packed QKV weights in {Function.Name} must be either three direct parameters or three canonical constants.");
            }
        }

        public void PropagateCalleeGroups(Call call, FunctionPlan calleePlan)
        {
            var arguments = call.Arguments.ToArray();
            foreach (var calleeGroup in calleePlan.ParameterGroups)
            {
                var weights = new[]
                {
                    RequireBuffer(arguments[calleeGroup.QIndex], "forwarded Q weight", call),
                    RequireBuffer(arguments[calleeGroup.KIndex], "forwarded K weight", call),
                    RequireBuffer(arguments[calleeGroup.VIndex], "forwarded V weight", call),
                };
                if (TryGetParameterTriple(weights, out var triple))
                {
                    AddGroup(triple, calleeGroup.Layout, $"call to {calleePlan.Function.Name}");
                }
                else if (!weights.All(IsCanonicalConstantBuffer))
                {
                    throw new InvalidOperationException(
                        $"Call from {Function.Name} to {calleePlan.Function.Name} mixes parameter and constant packed QKV weights.");
                }
            }
        }

        public bool TryFindGroup(IReadOnlyList<TIR.Buffer> weights, out ParameterGroup group)
        {
            if (TryGetParameterTriple(weights, out var triple) && _groups.TryGetValue(triple, out group!))
            {
                return true;
            }

            group = null!;
            return false;
        }

        private bool TryGetParameterTriple(IReadOnlyList<TIR.Buffer> weights, out ParameterTriple triple)
        {
            var indices = new int[3];
            for (var index = 0; index < weights.Count; index++)
            {
                if (weights[index].MemSpan.Buffer.Start is not BufferVar parameter ||
                    !_parameterIndices.TryGetValue(parameter, out indices[index]))
                {
                    triple = default;
                    return false;
                }
            }

            triple = new(indices[0], indices[1], indices[2]);
            return true;
        }

        private void AddGroup(ParameterTriple triple, FusedWeightLayout layout, string source)
        {
            if (triple.Q == triple.K || triple.Q == triple.V || triple.K == triple.V)
            {
                throw new InvalidOperationException(
                    $"Packed QKV parameter group in {Function.Name} contains duplicate parameters: {triple}.");
            }

            if (_groups.TryGetValue(triple, out var existing))
            {
                if (!existing.Layout.IsCompatibleWith(layout))
                {
                    throw new InvalidOperationException(
                        $"Packed QKV parameter group {triple} in {Function.Name} has incompatible layouts at {source}.");
                }

                return;
            }

            if (_groups.Values.Any(group => group.Indices.Intersect(new[] { triple.Q, triple.K, triple.V }).Any()))
            {
                throw new InvalidOperationException(
                    $"Packed QKV parameter groups overlap in {Function.Name}; each weight parameter must belong to one canonical group.");
            }

            _groups.Add(triple, new ParameterGroup(triple.Q, triple.K, triple.V, layout));
        }
    }

    private sealed class FunctionCloner : ExprCloner<Unit>
    {
        private readonly PrimFunction _function;
        private readonly FunctionPlan _plan;
        private readonly IReadOnlyDictionary<PrimFunction, FunctionPlan> _plans;
        private readonly IReadOnlyDictionary<PrimFunction, PrimFunction> _replacements;
        private readonly Dictionary<ParameterGroup, TIR.Buffer> _fusedParameterBuffers = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<DerivedWeightKey, TIR.Buffer> _derivedBuffers = new();
        private readonly HashSet<BufferVar> _removedParameters = new(ReferenceEqualityComparer.Instance);
        private int _bufferIndex;

        public FunctionCloner(
            PrimFunction function,
            FunctionPlan plan,
            IReadOnlyDictionary<PrimFunction, FunctionPlan> plans,
            IReadOnlyDictionary<PrimFunction, PrimFunction> replacements)
        {
            _function = function;
            _plan = plan;
            _plans = plans;
            _replacements = replacements;
        }

        public PrimFunction CloneFunction()
        {
            var groupByQ = _plan.ParameterGroups.ToDictionary(group => group.QIndex);
            var removedIndices = _plan.ParameterGroups
                .SelectMany(group => new[] { group.KIndex, group.VIndex })
                .ToHashSet();
            var parameters = new List<IVar>();
            for (var index = 0; index < _function.Parameters.Length; index++)
            {
                if (removedIndices.Contains(index))
                {
                    _removedParameters.Add((BufferVar)_function.Parameters[index]);
                    continue;
                }

                if (groupByQ.TryGetValue(index, out var group))
                {
                    var original = (BufferVar)_function.Parameters[index];
                    _removedParameters.Add(original);
                    var fusedParameter = new BufferVar(
                        $"{original.Name}_qkv_fused",
                        group.Layout.LocalTensorType,
                        BufferVarRole.Input,
                        MemoryLocation.Input,
                        BufferLayoutAnnotation.ExactStrided(group.Layout.Strides));
                    parameters.Add(fusedParameter);
                    var physical = new PhysicalBuffer(
                        group.Layout.AlignmentBytes,
                        fusedParameter,
                        group.Layout.SizeBytes,
                        MemoryLocation.Input);
                    var fusedBuffer = new TIR.Buffer(
                        $"{original.Name}_qkv_fused_input",
                        group.Layout.LocalTensorType.DType,
                        new MemSpan(physical),
                        group.Layout.Dimensions,
                        group.Layout.Strides,
                        distributedType: null);
                    _fusedParameterBuffers.Add(group, fusedBuffer);
                    continue;
                }

                var parameter = _function.Parameters[index];
                parameters.Add(parameter);
            }

            var body = (Sequential)Clone(_function.Body, default);
            var results = (Return)Clone(_function.Results, default);
            var replacement = new PrimFunction(
                _function.Name,
                _function.ModuleKind,
                body,
                results,
                parameters.ToArray())
            {
                Metadata = _function.Metadata.Clone(),
                Role = _function.Role,
                SchedResult = _function.SchedResult,
            };
            if (!CompilerServices.InferenceType(replacement))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after canonicalizing packed QKV weights in {_function.Name}: " +
                    CompilerServices.Print(replacement));
            }

            return replacement;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (expr is Call call && TryCloneContractCall(call, context, out var rewrittenCall))
            {
                return rewrittenCall;
            }

            if (expr is IVar variable)
            {
                if (variable is BufferVar bufferVar && _removedParameters.Contains(bufferVar))
                {
                    throw new InvalidOperationException(
                        $"Packed QKV weight parameter {bufferVar.Name} in {_function.Name} has a non-QKV use; " +
                        "the canonical fused ABI cannot preserve an independent reference.");
                }

                // PrimFunction results and parameter-backed views identify ABI
                // storage by reference. Only Q/K/V parameters replaced by the
                // fused contract may change identity.
                return (BaseExpr)variable;
            }

            return base.DispatchVisit(expr, context);
        }

        private bool TryCloneContractCall(Call expr, Unit context, out BaseExpr result)
        {
            if (expr.Target is TIR.NTT.PackedQKVParallelLinear op)
            {
                var arguments = expr.Arguments.ToArray();
                var weights = GetWeightBuffers(arguments, expr);
                var fusedWeight = GetFusedWeight(weights, _plan.QKVLayouts[expr]);
                result = TIR.F.NTT.PackedQKVParallelLinearFusedRhs(
                    (Expr)Clone(arguments[0], context),
                    fusedWeight,
                    (Expr)Clone(arguments[4], context),
                    (Expr)Clone(arguments[5], context),
                    (Expr)Clone(arguments[6], context),
                    (Expr)Clone(arguments[7], context),
                    (Expr)Clone(arguments[8], context),
                    (Expr)Clone(arguments[9], context),
                    (Expr)Clone(arguments[10], context),
                    (Expr)Clone(arguments[11], context),
                    (Expr)Clone(arguments[12], context),
                    (Expr)Clone(arguments[13], context),
                    (Expr)Clone(arguments[14], context),
                    (Expr)Clone(arguments[15], context),
                    op.NumHeads,
                    op.NumKvHeads,
                    op.RhsLayout,
                    op.OutputNVectorLaneCount,
                    op.QuantizationMode,
                    _plan.QKVLayouts[expr].ProjectionNCapacities);
                return true;
            }

            if (expr.Target is PrimFunction callee &&
                _plans.TryGetValue(callee, out var calleePlan) &&
                _replacements.TryGetValue(callee, out var replacement))
            {
                var arguments = expr.Arguments.ToArray();
                var groupsByQ = calleePlan.ParameterGroups.ToDictionary(group => group.QIndex);
                var removed = calleePlan.ParameterGroups
                    .SelectMany(group => new[] { group.KIndex, group.VIndex })
                    .ToHashSet();
                var rewritten = new List<BaseExpr>();
                for (var index = 0; index < arguments.Length; index++)
                {
                    if (removed.Contains(index))
                    {
                        continue;
                    }

                    if (groupsByQ.TryGetValue(index, out var group))
                    {
                        var weights = new[]
                        {
                            RequireBuffer(arguments[group.QIndex], "forwarded Q weight", expr),
                            RequireBuffer(arguments[group.KIndex], "forwarded K weight", expr),
                            RequireBuffer(arguments[group.VIndex], "forwarded V weight", expr),
                        };
                        rewritten.Add(GetFusedWeight(weights, group.Layout));
                    }
                    else
                    {
                        rewritten.Add(Clone(arguments[index], context));
                    }
                }

                result = expr.With(target: replacement, arguments: rewritten.ToArray());
                return true;
            }

            result = null!;
            return false;
        }

        private TIR.Buffer GetFusedWeight(IReadOnlyList<TIR.Buffer> weights, FusedWeightLayout layout)
        {
            if (_plan.TryFindGroup(weights, out var group))
            {
                return _fusedParameterBuffers[group];
            }

            var sources = weights
                .Select((weight, index) => GetConstantSource(weight, layout.SourceDistributedTypes[index]))
                .ToArray();
            var key = new DerivedWeightKey(sources[0].Tensor, sources[1].Tensor, sources[2].Tensor, sources[0].DistributedType, sources[1].DistributedType, sources[2].DistributedType);
            if (_derivedBuffers.TryGetValue(key, out var existing))
            {
                return existing;
            }

            var placeholder = new TensorConst(
                Tensor.Zeros(
                    layout.LocalTensorType.DType,
                    layout.Dimensions.Select(dimension => dimension.FixedValue).ToArray()));
            var materialization = new ConcatenatedDistributedTensorRDataMaterialization(
                layout.LocalTensorType,
                sources,
                layout.ConcatenationAxis);
            var physical = new PhysicalBuffer(
                layout.AlignmentBytes,
                IR.F.Buffer.AddressOf(placeholder),
                layout.SizeBytes,
                MemoryLocation.BlockLocalRdata,
                blockLocalRDataMaterialization: materialization);
            var buffer = new TIR.Buffer(
                $"packed_qkv_fused_weight_{_bufferIndex++}",
                layout.LocalTensorType.DType,
                new MemSpan(physical),
                layout.Dimensions,
                layout.Strides,
                distributedType: null);
            _derivedBuffers.Add(key, buffer);
            return buffer;
        }
    }

    private sealed class CallCollector : ExprWalker
    {
        public CallCollector()
            : base(visitOtherFunctions: false)
        {
        }

        public List<Call> Calls { get; } = new();

        protected override Unit VisitLeafCall(Call expr)
        {
            Calls.Add(expr);
            return base.VisitLeafCall(expr);
        }
    }

    private sealed record ParameterGroup(int QIndex, int KIndex, int VIndex, FusedWeightLayout Layout)
    {
        public int[] Indices => [QIndex, KIndex, VIndex];
    }

    private sealed record FusedWeightLayout(
        TensorType LocalTensorType,
        Dimension[] Dimensions,
        Dimension[] Strides,
        Dimension SizeBytes,
        int AlignmentBytes,
        int ConcatenationAxis,
        IRArray<DistributedType> SourceDistributedTypes,
        IRArray<long> ProjectionNCapacities)
    {
        public bool IsCompatibleWith(FusedWeightLayout other)
            => LocalTensorType.Equals(other.LocalTensorType) &&
                Dimensions.SequenceEqual(other.Dimensions) &&
                Strides.SequenceEqual(other.Strides) &&
                SizeBytes.Equals(other.SizeBytes) &&
                AlignmentBytes == other.AlignmentBytes &&
                ConcatenationAxis == other.ConcatenationAxis &&
                SourceDistributedTypes.SequenceEqual(other.SourceDistributedTypes) &&
                ProjectionNCapacities.SequenceEqual(other.ProjectionNCapacities);

        public static FusedWeightLayout Create(
            IReadOnlyList<TIR.Buffer> weights,
            IReadOnlyList<TIR.Buffer> outputs,
            IR.NTT.PackedMatMulRhsLayout rhsLayout,
            int outputNVectorLaneCount)
        {
            var localShapes = weights.Select(GetLocalCapacityShape).ToArray();
            if (localShapes.Any(shape => shape.Length != 2) || outputs.Any(output => GetLocalCapacityShape(output).Length != 2))
            {
                throw new InvalidOperationException(
                    "PyNTT canonical packed QKV weights require rank-2 local weights and outputs.");
            }

            var concatenationAxis = rhsLayout switch
            {
                IR.NTT.PackedMatMulRhsLayout.KMajor => 1,
                IR.NTT.PackedMatMulRhsLayout.NMajorKPacked => 0,
                _ => throw new InvalidOperationException(
                    $"PyNTT cannot canonicalize packed QKV layout {rhsLayout}."),
            };
            var reductionAxis = 1 - concatenationAxis;
            if (outputNVectorLaneCount <= 0 ||
                weights.Skip(1).Any(weight => weight.ElemType != weights[0].ElemType) ||
                localShapes.Skip(1).Any(shape => shape[reductionAxis] != localShapes[0][reductionAxis]))
            {
                throw new InvalidOperationException(
                    "PyNTT canonical packed QKV weights require a positive output N lane count, " +
                    "one dtype, and one physical K capacity.");
            }

            var sourceDistributedTypes = weights
                .Select(weight => weight.DistributedType ?? throw new InvalidOperationException(
                    "PyNTT canonical packed QKV weights require distributed source descriptors."))
                .ToArray();

            var dimensions = localShapes[0]
                .Select(extent => (Dimension)extent)
                .ToArray();
            dimensions[concatenationAxis] = checked(
                localShapes.Sum(shape => shape[concatenationAxis]));
            var localType = new TensorType(weights[0].ElemType, new RankedShape(dimensions));
            (var size, var strides) = TensorUtilities.GetTensorMaxSizeAndStridesExpr(localType, distributedType: null);
            var projectionNCapacities = outputs
                .Select(output => checked(
                    GetLocalCapacityShape(output)[^1] * GetVectorLaneCount(output.ElemType)))
                .ToArray();
            if (outputs.Any(output => GetVectorLaneCount(output.ElemType) != outputNVectorLaneCount))
            {
                throw new InvalidOperationException(
                    $"PyNTT canonical packed QKV outputs must use {outputNVectorLaneCount} N lanes.");
            }

            return new(
                localType,
                dimensions,
                strides,
                size,
                weights[0].ElemType.SizeInBytes,
                concatenationAxis,
                sourceDistributedTypes,
                projectionNCapacities);
        }
    }

    private readonly record struct ParameterTriple(int Q, int K, int V);

    private readonly record struct DerivedWeightKey(
        TensorConst Q,
        TensorConst K,
        TensorConst V,
        DistributedType QType,
        DistributedType KType,
        DistributedType VType);

    private static TIR.Buffer[] GetWeightBuffers(IReadOnlyList<BaseExpr> arguments, Call context)
        =>
        [
            RequireBuffer(arguments[1], "Q weight", context),
            RequireBuffer(arguments[2], "K weight", context),
            RequireBuffer(arguments[3], "V weight", context),
        ];

    private static TIR.Buffer RequireBuffer(BaseExpr expression, string name, Call context)
        => expression as TIR.Buffer ?? throw new InvalidOperationException(
            $"{context.Target.GetType().Name} {name} must be a TIR.Buffer, got {expression.GetType().Name}.");

    private static bool IsCanonicalConstantBuffer(TIR.Buffer buffer)
        => buffer.MemSpan.Start.Equals(Dimension.Zero) &&
            buffer.DistributedType is not null &&
            buffer.DistributedStorageKind == DistributedBufferStorageKind.CanonicalGlobal &&
            buffer.MemSpan.Buffer.Start is Call { Target: AddressOf } addressOf &&
            addressOf[AddressOf.Input] is TensorConst;

    private static DistributedTensorRDataSource GetConstantSource(
        TIR.Buffer buffer,
        DistributedType materializationType)
    {
        if (!IsCanonicalConstantBuffer(buffer) ||
            buffer.MemSpan.Buffer.Start is not Call { Target: AddressOf } addressOf ||
            addressOf[AddressOf.Input] is not TensorConst tensor)
        {
            throw new InvalidOperationException(
                $"Packed QKV buffer {buffer.Name} is not a zero-offset canonical distributed constant.");
        }

        if (!Equals(tensor.CheckedTensorType, materializationType.TensorType))
        {
            throw new InvalidOperationException(
                $"Packed QKV source {buffer.Name} has tensor type {tensor.CheckedTensorType}, but the " +
                $"callee-local materialization contract requires {materializationType.TensorType}.");
        }

        return new(tensor, materializationType);
    }

    private static long[] GetLocalCapacityShape(TIR.Buffer buffer)
    {
        if (buffer.DistributedType is not { } distributedType)
        {
            return CompilerServices.GetMaxShape(new RankedShape(buffer.Dimensions.ToArray()));
        }

        var descriptor = DistributedUtility.GetLocalShardDescriptor(
            distributedType,
            new int[distributedType.Placement.Rank],
            DistributedUtility.DivideFlags.MaxShape);
        return descriptor.LocalCapacityShape.ToValueArray();
    }

    private static int GetVectorLaneCount(DataType dataType)
        => dataType is VectorType vectorType
            ? checked(vectorType.Lanes.Aggregate(1, static (product, lane) => product * lane) *
                GetVectorLaneCount(vectorType.ElemType))
            : 1;
}
