// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Tensors;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Propagate packed tensor layouts across internal high-level function boundaries.
/// </summary>
public sealed class FunctionBoundaryLayoutPropagationPass : ModulePass
{
    private const int MaxIterations = 16;

    private enum BoundaryTransformKind
    {
        Pack,
        Unpack,
        Transpose,
        Bitcast,
        Boxing,
    }

    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var useConstShardedView = CompileSession.CompileOptions.TargetOptions is INTTTargetOptions targetOptions
            && Nncase.Passes.Distributed.AutoDistributedRewriter.SupportsConstShardedView(targetOptions);
        var specializer = new FunctionLayoutSpecializer(input, useConstShardedView);
        for (int iteration = 0; iteration < MaxIterations; iteration++)
        {
            var layouts = CollectLayouts(input);
            if (layouts.Count == 0)
            {
                return Task.FromResult(input);
            }

            var anyMutated = false;
            var functions = input.Functions.OfType<Function>().ToArray();
            foreach (var function in functions)
            {
                var rewriter = new CallBoundaryLayoutRewriter(layouts, specializer, useConstShardedView);
                rewriter.Rewrite(function);
                if (rewriter.IsMutated)
                {
                    anyMutated = true;
                    if (!CompilerServices.InferenceType(function))
                    {
                        throw new InvalidOperationException($"Type inference failed after propagating function boundary layouts in {function.Name}.");
                    }
                }
            }

            if (!anyMutated)
            {
                return Task.FromResult(input);
            }
        }

        var remainingLayoutMap = CollectLayouts(input);
        var remainingLayoutFunctions = remainingLayoutMap.Keys.ToArray();
        var remainingLayouts = string.Join(", ", remainingLayoutMap.Select(kv => $"{kv.Key.Name}: {kv.Value}"));
        var lastSignature = remainingLayoutFunctions.LastOrDefault() is { } last
            ? string.Join(", ", last.Parameters.ToArray().Select(parameter => $"{parameter.Name}: {((BaseExpr)parameter).CheckedType}"))
            : string.Empty;
        var lastBody = remainingLayoutFunctions.LastOrDefault() is { } lastBodyFunction ? CompilerServices.Print(lastBodyFunction.Body) : string.Empty;
        throw new InvalidOperationException($"Function boundary layout propagation did not converge within {MaxIterations} iterations. Remaining layout functions: {remainingLayouts}. Last signature: {lastSignature}. Last body: {lastBody}");
    }

    private static Dictionary<Function, FunctionBoundaryLayout> CollectLayouts(IRModule module)
    {
        var layouts = new Dictionary<Function, FunctionBoundaryLayout>(ReferenceEqualityComparer.Instance);
        foreach (var function in module.Functions.OfType<Function>())
        {
            if (ReferenceEquals(function, module.Entry) || function.IsEntry)
            {
                continue;
            }

            if (!HasFunctionCallUser(function))
            {
                continue;
            }

            if (FunctionBoundaryLayout.TryCreate(function) is { } layout)
            {
                layouts.Add(function, layout);
            }
        }

        return layouts;
    }

    private static bool HasFunctionCallUser(Function function)
    {
        foreach (var user in function.Users)
        {
            if (user is Call { Target: Function callTarget } && ReferenceEquals(callTarget, function))
            {
                return true;
            }

            if (user is FunctionWrapper { Target: Function wrapperTarget } && ReferenceEquals(wrapperTarget, function))
            {
                return true;
            }
        }

        return false;
    }

    private static bool TryStripInverseFromArgument(BaseExpr expr, BoundaryTransform transform, out BaseExpr transformed)
    {
        if (transform.TryStripInverse(expr, out transformed))
        {
            return true;
        }

        if (expr is Call { Target: GetItem, Arguments: var getItemArgs }
            && getItemArgs[GetItem.Input.Index] is IR.Tuple tuple
            && TryGetItemIndex(getItemArgs[GetItem.Index.Index], out var index)
            && index >= 0
            && index < tuple.Count)
        {
            return TryStripInverseFromArgument(tuple.Fields[index], transform, out transformed);
        }

        transformed = null!;
        return false;
    }

    private static bool TryGetItemIndex(BaseExpr expr, out int index)
    {
        switch (expr)
        {
            case DimConst dim:
                index = checked((int)dim.Value);
                return true;
            case TensorConst { Value: { Rank: 0 } tensor }:
                index = checked((int)tensor.ToScalar<long>());
                return true;
            default:
                index = -1;
                return false;
        }
    }

    private static BaseExpr MakeBoundaryTransform(Expr input, BoundaryTransform transform, bool useConstShardedView)
    {
        var expr = transform.Apply(input, useConstShardedView);
        Infer(expr, $"{transform.Kind} inserted at function boundary");
        return expr;
    }

    private static BaseExpr MakeInverseBoundaryTransform(Expr input, BoundaryTransform transform, IRType originalType)
    {
        var expr = transform.ApplyInverse(input, originalType);
        Infer(expr, $"inverse {transform.Kind} inserted for raw parameter use");
        return expr;
    }

    private static void Infer(BaseExpr expr, string context)
    {
        if (!CompilerServices.InferenceType(expr))
        {
            throw new InvalidOperationException($"Type inference failed for {context}.");
        }
    }

    private sealed class CallBoundaryLayoutRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<Function, FunctionBoundaryLayout> _layouts;
        private readonly FunctionLayoutSpecializer _specializer;
        private readonly bool _useConstShardedView;

        public CallBoundaryLayoutRewriter(
            IReadOnlyDictionary<Function, FunctionBoundaryLayout> layouts,
            FunctionLayoutSpecializer specializer,
            bool useConstShardedView)
            : base(visitOtherFunctions: false)
        {
            _layouts = layouts;
            _specializer = specializer;
            _useConstShardedView = useConstShardedView;
        }

        protected override BaseExpr RewriteLeafCall(Call expr)
        {
            if (TryFoldPackUnpack(expr, out var folded))
            {
                return folded;
            }

            if (!TryGetTargetFunction(expr.Target, out var function, out var wrapper)
                || !_layouts.TryGetValue(function, out var layout))
            {
                return expr;
            }

            var specialized = _specializer.GetOrCreate(function, layout);
            Expr target = wrapper is null ? specialized : wrapper.With(target: specialized);
            Infer(target, $"specialized target {specialized.Name}");

            var args = expr.Arguments.ToArray();
            for (int i = 0; i < layout.Inputs.Length; i++)
            {
                var inputLayout = layout.Inputs[i];
                if (inputLayout is null)
                {
                    continue;
                }

                if (TryStripInverseFromArgument(args[i], inputLayout.Transform, out var transformedArg))
                {
                    args[i] = transformedArg;
                    continue;
                }

                if (args[i] is not Expr argExpr)
                {
                    throw new InvalidOperationException($"Cannot pack non-expression argument {i} for call to {function.Name}.");
                }

                args[i] = MakeBoundaryTransform(argExpr, inputLayout.Transform, _useConstShardedView);
            }

            var rawCall = expr.With(target: target, arguments: args);
            Infer(rawCall, $"call to specialized function {specialized.Name}");

            return WrapOutputs(rawCall, layout);
        }

        private static bool TryGetTargetFunction(Expr target, out Function function, out FunctionWrapper? wrapper)
        {
            switch (target)
            {
                case Function fn:
                    function = fn;
                    wrapper = null;
                    return true;
                case FunctionWrapper { ReturnOutput: true, Target: Function fn } fw:
                    function = fn;
                    wrapper = fw;
                    return true;
                default:
                    function = null!;
                    wrapper = null;
                    return false;
            }
        }

        private static bool TryFoldPackUnpack(Call expr, out BaseExpr folded)
        {
            if (expr.Target is Pack pack
                && BoundaryTransform.Pack(pack).TryStripInverse(expr.Arguments[Pack.Input.Index], out folded))
            {
                return true;
            }

            if (expr.Target is Unpack unpack
                && BoundaryTransform.Unpack(unpack).TryStripInverse(expr.Arguments[Unpack.Input.Index], out folded))
            {
                return true;
            }

            folded = null!;
            return false;
        }

        private BaseExpr WrapOutputs(Call rawCall, FunctionBoundaryLayout layout)
        {
            if (!layout.HasOutputLayout)
            {
                return rawCall;
            }

            if (rawCall.CheckedType is not TupleType)
            {
                var outputLayout = layout.Outputs[0] ?? throw new InvalidOperationException("Single-output call has no output layout to wrap.");
                return MakeBoundaryTransform(rawCall, outputLayout.Transform, _useConstShardedView);
            }

            var fields = new BaseExpr[layout.Outputs.Length];
            for (int i = 0; i < fields.Length; i++)
            {
                BaseExpr field = IR.F.Tensors.GetItem(rawCall, i);
                Infer(field, $"tuple field {i} from {rawCall.Target}");
                if (layout.Outputs[i] is { } outputLayout)
                {
                    if (field is not Expr fieldExpr)
                    {
                        throw new InvalidOperationException($"Cannot unpack non-expression output {i} from call to {rawCall.Target}.");
                    }

                    field = MakeBoundaryTransform(fieldExpr, outputLayout.Transform, _useConstShardedView);
                }

                fields[i] = field;
            }

            var tuple = new IR.Tuple(fields);
            Infer(tuple, "tuple output wrapper");
            return tuple;
        }
    }

    private sealed class FunctionLayoutSpecializer
    {
        private readonly IRModule _module;
        private readonly bool _useConstShardedView;
        private readonly Dictionary<Function, Dictionary<FunctionBoundaryLayout, Function>> _cache = new(ReferenceEqualityComparer.Instance);
        private int _nextId;

        public FunctionLayoutSpecializer(IRModule module, bool useConstShardedView)
        {
            _module = module;
            _useConstShardedView = useConstShardedView;
        }

        public Function GetOrCreate(Function function, FunctionBoundaryLayout layout)
        {
            if (!_cache.TryGetValue(function, out var perFunction))
            {
                perFunction = new Dictionary<FunctionBoundaryLayout, Function>();
                _cache.Add(function, perFunction);
            }

            if (perFunction.TryGetValue(layout, out var existing))
            {
                return existing;
            }

            var specialized = Create(function, layout);
            perFunction.Add(layout, specialized);
            _module.Add(specialized);
            return specialized;
        }

        private Function Create(Function function, FunctionBoundaryLayout layout)
        {
            var parameters = function.Parameters.ToArray();
            var mappedParameters = new Dictionary<Var, Var>(ReferenceEqualityComparer.Instance);
            var specializedParameters = new IVar[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is not Var parameter)
                {
                    if (layout.Inputs[i] is not null)
                    {
                        throw new InvalidOperationException($"Function boundary layout unexpectedly contains a tensor layout for non-Var parameter {i} in {function.Name}.");
                    }

                    specializedParameters[i] = parameters[i];
                    continue;
                }

                var type = layout.Inputs[i]?.TransformedType ?? parameter.CheckedType;
                var specializedParameter = parameter.With(typeAnnotation: type);
                mappedParameters.Add(parameter, specializedParameter);
                specializedParameters[i] = specializedParameter;
            }

            var packedInputs = new Dictionary<Var, PortLayout>(ReferenceEqualityComparer.Instance);
            for (int i = 0; i < layout.Inputs.Length; i++)
            {
                if (layout.Inputs[i] is { } inputLayout)
                {
                    packedInputs.Add((Var)parameters[i], inputLayout);
                }
            }

            var cloner = new SpecializedBodyCloner(mappedParameters, packedInputs, _useConstShardedView);
            var body = CloneBodyWithPackedOutputs(function.Body, layout, cloner);
            var varMap = RewriteVarMap(function.VarMap, mappedParameters);
            var specialized = new Function($"{function.Name}__layout_{_nextId++}", function.ModuleKind, body, specializedParameters, varMap);
            if (!CompilerServices.InferenceType(specialized))
            {
                throw new InvalidOperationException($"Type inference failed for specialized function {specialized.Name} generated from {function.Name}.");
            }

            return specialized;
        }

        private BaseExpr CloneBodyWithPackedOutputs(BaseExpr body, FunctionBoundaryLayout layout, SpecializedBodyCloner cloner)
        {
            var fields = FunctionBoundaryLayout.FlattenReturn(body);
            if (fields.Length != layout.Outputs.Length)
            {
                throw new InvalidOperationException("Function output layout metadata is out of sync with function body.");
            }

            var clonedFields = new BaseExpr[fields.Length];
            for (int i = 0; i < fields.Length; i++)
            {
                var source = fields[i];
                if (layout.Outputs[i] is { } outputLayout
                    && outputLayout.Transform.TryStrip(source, out var transformedOutput))
                {
                    source = transformedOutput;
                }

                clonedFields[i] = cloner.Clone(source, default);
            }

            if (body is IR.Tuple)
            {
                var tuple = new IR.Tuple(clonedFields);
                Infer(tuple, "specialized tuple body");
                return tuple;
            }

            return clonedFields[0];
        }

        private Dictionary<IVar, Dimension[]> RewriteVarMap(Dictionary<IVar, Dimension[]>? source, IReadOnlyDictionary<Var, Var> mappedParameters)
        {
            var result = new Dictionary<IVar, Dimension[]>();
            if (source is null)
            {
                return result;
            }

            foreach (var (key, value) in source)
            {
                if (key is Var var && mappedParameters.TryGetValue(var, out var mapped))
                {
                    result[mapped] = value;
                }
                else
                {
                    result[key] = value;
                }
            }

            return result;
        }
    }

    private sealed class SpecializedBodyCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<Var, Var> _mappedParameters;
        private readonly Dictionary<Var, PortLayout> _packedInputs;
        private readonly bool _useConstShardedView;

        public SpecializedBodyCloner(IReadOnlyDictionary<Var, Var> mappedParameters, Dictionary<Var, PortLayout> packedInputs, bool useConstShardedView)
            : base(cloneOtherFunctions: false)
        {
            _mappedParameters = mappedParameters;
            _packedInputs = packedInputs;
            _useConstShardedView = useConstShardedView;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (expr is Call call
                && BoundaryTransform.TryCreate(call, out var transform, out var input)
                && input is Var parameter
                && _packedInputs.TryGetValue(parameter, out var inputLayout))
            {
                if (inputLayout.Transform.Equals(transform))
                {
                    return _mappedParameters[parameter];
                }

                if (inputLayout.Transform.IsDistributedBoxing
                    && transform.IsDistributedBoxing)
                {
                    return MakeBoundaryTransform(_mappedParameters[parameter], transform, _useConstShardedView);
                }
            }

            return base.DispatchVisit(expr, context);
        }

        protected override BaseExpr VisitLeafVar(Var expr, Unit context)
        {
            if (!_mappedParameters.TryGetValue(expr, out var mapped))
            {
                return base.VisitLeafVar(expr, context);
            }

            if (_packedInputs.TryGetValue(expr, out var inputLayout))
            {
                return MakeInverseBoundaryTransform(mapped, inputLayout.Transform, expr.CheckedType);
            }

            return mapped;
        }

        protected override BaseExpr VisitLeafDimVar(DimVar expr, Unit context) => expr;
    }

    private sealed class FunctionBoundaryLayout : IEquatable<FunctionBoundaryLayout>
    {
        public FunctionBoundaryLayout(PortLayout?[] inputs, PortLayout?[] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        public PortLayout?[] Inputs { get; }

        public PortLayout?[] Outputs { get; }

        public bool HasOutputLayout => Outputs.Any(x => x is not null);

        public static FunctionBoundaryLayout? TryCreate(Function function)
        {
            var inputs = AnalyzeInputs(function);
            var outputs = AnalyzeOutputs(function.Body);
            if (inputs.All(x => x is null) && outputs.All(x => x is null))
            {
                return null;
            }

            return new FunctionBoundaryLayout(inputs, outputs);
        }

        public static BaseExpr[] FlattenReturn(BaseExpr body) => body is IR.Tuple tuple ? tuple.Fields.ToArray() : [body];

        public bool Equals(FunctionBoundaryLayout? other)
        {
            return other is not null
                && LayoutArraysEqual(Inputs, other.Inputs)
                && LayoutArraysEqual(Outputs, other.Outputs);
        }

        public override bool Equals(object? obj) => Equals(obj as FunctionBoundaryLayout);

        public override int GetHashCode()
        {
            var hash = default(HashCode);
            AddLayoutArrayHash(ref hash, Inputs);
            AddLayoutArrayHash(ref hash, Outputs);
            return hash.ToHashCode();
        }

        public override string ToString()
        {
            return $"inputs=[{string.Join(", ", Inputs.Select(x => x?.ToString() ?? "-"))}], outputs=[{string.Join(", ", Outputs.Select(x => x?.ToString() ?? "-"))}]";
        }

        private static PortLayout?[] AnalyzeInputs(Function function)
        {
            var parameters = function.Parameters.ToArray();
            var result = new PortLayout?[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is not Var parameter)
                {
                    continue;
                }

                if (parameter.CheckedType is DistributedType)
                {
                    continue;
                }

                var candidates = new List<PortLayout>();
                foreach (var user in parameter.Users)
                {
                    if (ReferenceEquals(user, function))
                    {
                        continue;
                    }

                    if (user is Call call
                        && BoundaryTransform.TryCreateInputTransform(call, out var transform, out var input)
                        && ReferenceEquals(input, parameter))
                    {
                        candidates.Add(new PortLayout(transform, user.CheckedType));
                    }
                }

                result[i] = SelectCanonicalInputLayout(candidates);
            }

            return result;
        }

        private static PortLayout? SelectCanonicalInputLayout(IReadOnlyList<PortLayout> candidates)
        {
            if (candidates.Count == 0)
            {
                return null;
            }

            var first = candidates[0];
            if (candidates.All(candidate => candidate.Transform.Equals(first.Transform)))
            {
                return first;
            }

            if (!candidates.All(candidate => candidate.Transform.IsDistributedBoxing))
            {
                return null;
            }

            return candidates
                .GroupBy(candidate => candidate.Transform)
                .OrderByDescending(group => group.Count())
                .ThenBy(group => group.Key.ToString(), StringComparer.Ordinal)
                .Select(group => group.First())
                .First();
        }

        private static PortLayout?[] AnalyzeOutputs(BaseExpr body)
        {
            var fields = FlattenReturn(body);
            var result = new PortLayout?[fields.Length];
            for (int i = 0; i < fields.Length; i++)
            {
                if (fields[i] is Call call
                    && BoundaryTransform.TryCreateOutputTransform(call, out var transform, out var input))
                {
                    result[i] = new PortLayout(transform, input.CheckedType);
                }
            }

            return result;
        }

        private static bool LayoutArraysEqual(IReadOnlyList<PortLayout?> lhs, IReadOnlyList<PortLayout?> rhs)
        {
            if (lhs.Count != rhs.Count)
            {
                return false;
            }

            for (int i = 0; i < lhs.Count; i++)
            {
                var lhsLayout = lhs[i]?.Transform;
                var rhsLayout = rhs[i]?.Transform;
                if (lhsLayout is null || rhsLayout is null)
                {
                    if (lhsLayout is not null || rhsLayout is not null)
                    {
                        return false;
                    }

                    continue;
                }

                if (!lhsLayout.Equals(rhsLayout))
                {
                    return false;
                }
            }

            return true;
        }

        private static void AddLayoutArrayHash(ref HashCode hash, IEnumerable<PortLayout?> values)
        {
            foreach (var value in values)
            {
                hash.Add(value?.Transform);
            }
        }
    }

    private sealed class PortLayout
    {
        public PortLayout(BoundaryTransform transform, IRType transformedType)
        {
            Transform = transform;
            TransformedType = transformedType;
        }

        public BoundaryTransform Transform { get; }

        public IRType TransformedType { get; }

        public override string ToString() => $"{Transform}->{TransformedType}";
    }

    private sealed class BoundaryTransform : IEquatable<BoundaryTransform>
    {
        private BoundaryTransform(BoundaryTransformKind kind, IReadOnlyList<int>? lanes = null, IReadOnlyList<int>? axes = null, IReadOnlyList<int>? perm = null, DataType? newType = null, IRType? newIRType = null)
        {
            Kind = kind;
            Lanes = lanes?.ToArray() ?? Array.Empty<int>();
            Axes = axes?.ToArray() ?? Array.Empty<int>();
            Perm = perm?.ToArray() ?? Array.Empty<int>();
            NewType = newType;
            NewIRType = newIRType;
        }

        public BoundaryTransformKind Kind { get; }

        public int[] Lanes { get; }

        public int[] Axes { get; }

        public int[] Perm { get; }

        public DataType? NewType { get; }

        public IRType? NewIRType { get; }

        public bool IsDistributedBoxing => Kind is BoundaryTransformKind.Boxing && NewIRType is DistributedType;

        public static BoundaryTransform Pack(Pack pack) => new(BoundaryTransformKind.Pack, lanes: pack.Lanes.ToArray(), axes: pack.Axes.ToArray());

        public static BoundaryTransform Unpack(Unpack unpack) => new(BoundaryTransformKind.Unpack, lanes: unpack.Lanes.ToArray(), axes: unpack.Axes.ToArray());

        public static BoundaryTransform Transpose(IReadOnlyList<int> perm) => new(BoundaryTransformKind.Transpose, perm: perm);

        public static BoundaryTransform Bitcast(Bitcast bitcast) => new(BoundaryTransformKind.Bitcast, newType: bitcast.NewType);

        public static BoundaryTransform Boxing(IR.Distributed.Boxing boxing) => new(BoundaryTransformKind.Boxing, newIRType: boxing.NewType);

        public static bool TryCreate(Call call, out BoundaryTransform transform, out BaseExpr input)
        {
            switch (call.Target)
            {
                case IR.Tensors.Pack pack:
                    transform = Pack(pack);
                    input = call.Arguments[IR.Tensors.Pack.Input.Index];
                    return true;
                case IR.Tensors.Unpack unpack:
                    transform = Unpack(unpack);
                    input = call.Arguments[IR.Tensors.Unpack.Input.Index];
                    return true;
                case IR.Tensors.Transpose when TryGetFixedPerm(call.Arguments[IR.Tensors.Transpose.Perm.Index], out var perm):
                    transform = Transpose(perm);
                    input = call.Arguments[IR.Tensors.Transpose.Input.Index];
                    return true;
                case IR.Tensors.Bitcast bitcast:
                    transform = Bitcast(bitcast);
                    input = call.Arguments[IR.Tensors.Bitcast.Input.Index];
                    return true;
                case IR.Distributed.Boxing boxing:
                    transform = Boxing(boxing);
                    input = call.Arguments[IR.Distributed.Boxing.Input.Index];
                    return true;
                default:
                    transform = null!;
                    input = null!;
                    return false;
            }
        }

        public static bool TryCreateInputTransform(Call call, out BoundaryTransform transform, out BaseExpr input)
        {
            if (TryCreate(call, out transform, out input)
                && (transform.Kind is BoundaryTransformKind.Pack or BoundaryTransformKind.Transpose
                    || (transform.Kind is BoundaryTransformKind.Boxing && transform.NewIRType is DistributedType)))
            {
                return true;
            }

            transform = null!;
            input = null!;
            return false;
        }

        public static bool TryCreateOutputTransform(Call call, out BoundaryTransform transform, out BaseExpr input)
        {
            if (TryCreate(call, out transform, out input)
                && transform.Kind is BoundaryTransformKind.Unpack or BoundaryTransformKind.Bitcast or BoundaryTransformKind.Transpose or BoundaryTransformKind.Boxing)
            {
                return true;
            }

            transform = null!;
            input = null!;
            return false;
        }

        public BaseExpr Apply(Expr input, bool useConstShardedView)
        {
            if (Kind is BoundaryTransformKind.Boxing)
            {
                var targetType = NewIRType ?? throw new InvalidOperationException("Boxing transform is missing target IR type.");
                if (Equals(input.CheckedType, targetType))
                {
                    return input;
                }

                if (useConstShardedView && input is TensorConst && targetType is DistributedType distributedType)
                {
                    return IR.F.Distributed.ShardedView(input, distributedType);
                }
            }

            return Kind switch
            {
                BoundaryTransformKind.Pack => IR.F.Tensors.Pack(input, Lanes, Axes),
                BoundaryTransformKind.Unpack => IR.F.Tensors.Unpack(input, Lanes, Axes),
                BoundaryTransformKind.Transpose => IR.F.Tensors.Transpose(input, Perm),
                BoundaryTransformKind.Bitcast => IR.F.Tensors.Bitcast(input, NewType ?? throw new InvalidOperationException("Bitcast transform is missing new dtype.")),
                BoundaryTransformKind.Boxing => IR.F.Distributed.Boxing(input, NewIRType ?? throw new InvalidOperationException("Boxing transform is missing target IR type.")),
                _ => throw new InvalidOperationException($"Unsupported boundary transform kind: {Kind}."),
            };
        }

        public BaseExpr ApplyInverse(Expr input, IRType originalType)
        {
            return Kind switch
            {
                BoundaryTransformKind.Pack => IR.F.Tensors.Unpack(input, Lanes, Axes),
                BoundaryTransformKind.Unpack => IR.F.Tensors.Pack(input, Lanes, Axes),
                BoundaryTransformKind.Transpose => IR.F.Tensors.Transpose(input, InvertPerm(Perm)),
                BoundaryTransformKind.Bitcast => IR.F.Tensors.Bitcast(input, GetDType(originalType, "bitcast inverse transform")),
                BoundaryTransformKind.Boxing => IR.F.Distributed.Boxing(input, originalType),
                _ => throw new InvalidOperationException($"Unsupported boundary transform kind: {Kind}."),
            };
        }

        public bool TryStrip(BaseExpr expr, out BaseExpr input)
        {
            if (expr is Call call
                && TryCreate(call, out var transform, out input)
                && Equals(transform))
            {
                return true;
            }

            input = null!;
            return false;
        }

        public bool TryStripInverse(BaseExpr expr, out BaseExpr input)
        {
            if (expr is not Call call || !TryCreate(call, out var transform, out input))
            {
                input = null!;
                return false;
            }

            if (Kind == BoundaryTransformKind.Boxing
                && transform.Kind == BoundaryTransformKind.Boxing
                && Equals(input.CheckedType, NewIRType))
            {
                return true;
            }

            var inverse = Kind switch
            {
                BoundaryTransformKind.Pack when transform.Kind == BoundaryTransformKind.Unpack => new BoundaryTransform(BoundaryTransformKind.Pack, lanes: transform.Lanes, axes: transform.Axes),
                BoundaryTransformKind.Unpack when transform.Kind == BoundaryTransformKind.Pack => new BoundaryTransform(BoundaryTransformKind.Unpack, lanes: transform.Lanes, axes: transform.Axes),
                BoundaryTransformKind.Transpose when transform.Kind == BoundaryTransformKind.Transpose => Transpose(InvertPerm(transform.Perm)),
                _ => null,
            };

            if (Equals(inverse))
            {
                return true;
            }

            input = null!;
            return false;
        }

        public bool Equals(BoundaryTransform? other)
        {
            return other is not null
                && Kind == other.Kind
                && Lanes.SequenceEqual(other.Lanes)
                && Axes.SequenceEqual(other.Axes)
                && Perm.SequenceEqual(other.Perm)
                && Equals(NewType, other.NewType)
                && Equals(NewIRType, other.NewIRType);
        }

        public override bool Equals(object? obj) => Equals(obj as BoundaryTransform);

        public override int GetHashCode()
        {
            var hash = default(HashCode);
            hash.Add(Kind);
            foreach (var lane in Lanes)
            {
                hash.Add(lane);
            }

            hash.Add(-1);
            foreach (var axis in Axes)
            {
                hash.Add(axis);
            }

            hash.Add(-2);
            foreach (var item in Perm)
            {
                hash.Add(item);
            }

            hash.Add(NewType);
            hash.Add(NewIRType);
            return hash.ToHashCode();
        }

        public override string ToString()
        {
            return Kind switch
            {
                BoundaryTransformKind.Pack => $"Pack([{string.Join(",", Lanes)}], [{string.Join(",", Axes)}])",
                BoundaryTransformKind.Unpack => $"Unpack([{string.Join(",", Lanes)}], [{string.Join(",", Axes)}])",
                BoundaryTransformKind.Transpose => $"Transpose([{string.Join(",", Perm)}])",
                BoundaryTransformKind.Bitcast => $"Bitcast({NewType})",
                BoundaryTransformKind.Boxing => $"Boxing({NewIRType})",
                _ => Kind.ToString(),
            };
        }

        private static bool TryGetFixedPerm(BaseExpr expr, out int[] perm)
        {
            if (expr is Shape { IsFixed: true } shape)
            {
                perm = shape.ToValueArray().Select(value => checked((int)value)).ToArray();
                return true;
            }

            perm = Array.Empty<int>();
            return false;
        }

        private static int[] InvertPerm(IReadOnlyList<int> perm)
        {
            var inverse = new int[perm.Count];
            for (int i = 0; i < perm.Count; i++)
            {
                inverse[perm[i]] = i;
            }

            return inverse;
        }

        private static DataType GetDType(IRType type, string context)
        {
            return type switch
            {
                TensorType tensor => tensor.DType,
                DistributedType distributed => distributed.TensorType.DType,
                _ => throw new InvalidOperationException($"Cannot get tensor dtype for {context} from {type}."),
            };
        }
    }
}
