// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Schedule;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Inlines prim functions that have exactly one static call site in a target module.
/// </summary>
public sealed class InlineSingleCallPrimFunctionsPass : ModulePass
{
    private readonly string _moduleKind;

    public InlineSingleCallPrimFunctionsPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    /// <inheritdoc/>
    protected override async Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var targetFunctions = input.Functions
            .OfType<PrimFunction>()
            .Where(function => function.ModuleKind == _moduleKind)
            .ToHashSet(new ReferenceEqualityComparer<PrimFunction>());
        if (targetFunctions.Count == 0)
        {
            return input;
        }

        var callSites = targetFunctions.ToDictionary(
            function => function,
            _ => new List<PrimFunctionCallSite>(),
            new ReferenceEqualityComparer<PrimFunction>());
        foreach (var caller in input.Functions)
        {
            var collector = new PrimFunctionCallSiteCollector(caller, targetFunctions, _moduleKind, callSites);
            collector.Visit(caller);
        }

        var inlineFunctions = new HashSet<PrimFunction>(new ReferenceEqualityComparer<PrimFunction>());
        foreach (var (callee, sites) in callSites)
        {
            if (sites.Count != 1 || callee.Role == FunctionRole.Dispatch)
            {
                continue;
            }

            var site = sites[0];
            if (site.Caller is not PrimFunction caller
                || caller.ModuleKind != _moduleKind
                || caller.Role == FunctionRole.Dispatch)
            {
                continue;
            }

            if (site.Parent is not Sequential)
            {
                throw new InvalidOperationException(
                    $"Cannot inline prim function {callee.Name} into {caller.Name}: " +
                    "prim function calls must be direct TIR.Sequential statements.");
            }

            inlineFunctions.Add(callee);
        }

        if (inlineFunctions.Count == 0)
        {
            return input;
        }

        foreach (var function in targetFunctions)
        {
            var rewriter = new PrimFunctionCallInliner(inlineFunctions, function);
            rewriter.Rewrite(function);
            if (!rewriter.IsMutated)
            {
                continue;
            }

            if (!CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after inlining prim function calls in {function.Name}: {DescribeInvalidTypes(function)}");
            }
        }

        await new RemoveUnusedFunctions(CompileSession.CompileOptions).RunAsync(input, context);
        return input;
    }

    private static string DescribeInvalidTypes(PrimFunction function)
    {
        var invalidExpressions = ExprCollector.Collect(function)
            .Select(expr => (Expr: expr, Type: IRHelpers.GetRawCheckedType(expr)))
            .Where(item => item.Type is InvalidType)
            .Take(8)
            .Select(item => $"{DescribeExpression(item.Expr)} => {((InvalidType)item.Type!).Reason}")
            .ToArray();
        return invalidExpressions.Length == 0
            ? "no InvalidType node was retained"
            : string.Join("; ", invalidExpressions);
    }

    private static string DescribeExpression(BaseExpr expr)
        => expr switch
        {
            Call { Target: BaseFunction function } => $"Call({function.Name})",
            Call { Target: Op op } => $"Call({op.GetType().Name})",
            BaseFunction function => $"{expr.GetType().Name}({function.Name})",
            _ => expr.GetType().Name,
        };

    private sealed record PrimFunctionCallSite(BaseFunction Caller, BaseExpr? Parent);

    private sealed class PrimFunctionCallSiteCollector : ExprWalker
    {
        private readonly BaseFunction _caller;
        private readonly IReadOnlySet<PrimFunction> _targetFunctions;
        private readonly string _moduleKind;
        private readonly IReadOnlyDictionary<PrimFunction, List<PrimFunctionCallSite>> _callSites;
        private BaseExpr? _parent;

        public PrimFunctionCallSiteCollector(
            BaseFunction caller,
            IReadOnlySet<PrimFunction> targetFunctions,
            string moduleKind,
            IReadOnlyDictionary<PrimFunction, List<PrimFunctionCallSite>> callSites)
            : base(visitOtherFunctions: false)
        {
            _caller = caller;
            _targetFunctions = targetFunctions;
            _moduleKind = moduleKind;
            _callSites = callSites;
        }

        protected override void VisitOperands(BaseExpr expr, Unit context)
        {
            foreach (var operand in expr.Operands)
            {
                var previousParent = _parent;
                _parent = expr;
                Visit(operand, context);
                _parent = previousParent;
            }
        }

        protected override Unit VisitLeafCall(Call expr)
        {
            if (expr.Target is not PrimFunction callee || callee.ModuleKind != _moduleKind)
            {
                return default;
            }

            if (!_targetFunctions.Contains(callee))
            {
                throw new InvalidOperationException(
                    $"Prim function {_caller.Name} calls {callee.Name}, but the callee is not registered in the IR module.");
            }

            _callSites[callee].Add(new PrimFunctionCallSite(_caller, _parent));
            return default;
        }
    }

    private sealed class PrimFunctionCallInliner : ExprRewriter
    {
        private readonly IReadOnlySet<PrimFunction> _inlineFunctions;
        private readonly IReadOnlySet<PrimFunction> _activeFunctions;
        private readonly PrimFunction _destinationFunction;
        private readonly HashSet<PrimFunction> _mergedScheduleFunctions;

        public PrimFunctionCallInliner(IReadOnlySet<PrimFunction> inlineFunctions, PrimFunction root)
            : this(
                inlineFunctions,
                new HashSet<PrimFunction>(new ReferenceEqualityComparer<PrimFunction>()) { root },
                root,
                new HashSet<PrimFunction>(new ReferenceEqualityComparer<PrimFunction>()))
        {
        }

        private PrimFunctionCallInliner(
            IReadOnlySet<PrimFunction> inlineFunctions,
            IReadOnlySet<PrimFunction> activeFunctions,
            PrimFunction destinationFunction,
            HashSet<PrimFunction> mergedScheduleFunctions)
            : base(visitOtherFunctions: false)
        {
            _inlineFunctions = inlineFunctions;
            _activeFunctions = activeFunctions;
            _destinationFunction = destinationFunction;
            _mergedScheduleFunctions = mergedScheduleFunctions;
        }

        protected override BaseExpr RewriteLeafCall(Call expr)
        {
            if (expr.Target is not PrimFunction callee || !_inlineFunctions.Contains(callee))
            {
                return expr;
            }

            if (_activeFunctions.Contains(callee))
            {
                throw new InvalidOperationException($"Recursive prim function call detected while inlining {callee.Name}.");
            }

            var parameters = callee.Parameters.ToArray();
            var arguments = expr.Arguments.ToArray();
            if (parameters.Length != arguments.Length)
            {
                throw new InvalidOperationException(
                    $"Cannot inline prim function {callee.Name}: expected {parameters.Length} arguments, got {arguments.Length}.");
            }

            MergeReadOnlyDataAllocations(callee);

            var replacements = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
            var workspaceBases = new Dictionary<MemoryLocation, TIR.Buffer>();
            for (int i = 0; i < parameters.Length; i++)
            {
                replacements.Add((BaseExpr)parameters[i], arguments[i]);
                if (parameters[i] is BufferVar { Role: BufferVarRole.Workspace } workspace)
                {
                    if (arguments[i] is not TIR.Buffer actualWorkspace)
                    {
                        throw new InvalidOperationException(
                            $"Cannot inline prim function {callee.Name}: workspace {workspace.Name} " +
                            $"must be backed by a TIR.Buffer, got {arguments[i].GetType().Name}.");
                    }

                    if (!workspaceBases.TryAdd(workspace.Location, actualWorkspace))
                    {
                        throw new InvalidOperationException(
                            $"Cannot inline prim function {callee.Name}: multiple workspaces use memory location {workspace.Location}.");
                    }
                }
            }

            var clonedBody = new PrimFunctionBodyCloner(replacements, workspaceBases).Clone(callee.Body, default);
            var activeFunctions = new HashSet<PrimFunction>(_activeFunctions, new ReferenceEqualityComparer<PrimFunction>())
            {
                callee,
            };
            var nestedInliner = new PrimFunctionCallInliner(
                _inlineFunctions,
                activeFunctions,
                _destinationFunction,
                _mergedScheduleFunctions);
            var inlinedBody = (Sequential)nestedInliner.Rewrite(clonedBody);
            inlinedBody = inlinedBody.With(
                traceScopeName: inlinedBody.TraceScopeName ?? callee.Name,
                preserveCodegenBoundary: true);

            SetMutated();
            return inlinedBody;
        }

        protected override BaseExpr RewriteLeafSequential(Sequential expr)
        {
            foreach (var field in expr.Fields)
            {
                if (field is Sequential { CanFlatten: true })
                {
                    SetMutated();
                    var flattened = Sequential.Flatten(expr.Fields);
                    return expr.CanFlatten
                        ? flattened
                        : flattened.With(
                            traceScopeName: expr.TraceScopeName,
                            preserveCodegenBoundary: expr.PreserveCodegenBoundary);
                }
            }

            return expr;
        }

        private static void MergeReadOnlyDataAllocations(
            IDictionary<Const, ValueRange<ulong>> destination,
            IReadOnlyDictionary<Const, ValueRange<ulong>> source,
            PrimFunction destinationFunction,
            PrimFunction sourceFunction,
            MemoryLocation location)
        {
            foreach (var (@const, range) in source)
            {
                var allocationKey = @const;
                if (destination.TryGetValue(allocationKey, out var existingRange))
                {
                    if (existingRange == range)
                    {
                        continue;
                    }

                    allocationKey = CloneReadOnlyDataConstant(@const, destinationFunction, sourceFunction, location);
                }

                destination.Add(allocationKey, range);
            }
        }

        private static Const CloneReadOnlyDataConstant(
            Const @const,
            PrimFunction destinationFunction,
            PrimFunction sourceFunction,
            MemoryLocation location)
            => @const switch
            {
                TensorConst tensorConst => tensorConst.With(),
                _ => throw new NotSupportedException(
                    $"Cannot inline readonly {location} allocation from {sourceFunction.Name} into " +
                    $"{destinationFunction.Name}: expected TensorConst, got {@const.GetType().Name}."),
            };

        private void MergeReadOnlyDataAllocations(PrimFunction callee)
        {
            if (!_mergedScheduleFunctions.Add(callee))
            {
                return;
            }

            MergeReadOnlyDataAllocations(
                _destinationFunction.SchedResult.Rdatas,
                callee.SchedResult.Rdatas,
                _destinationFunction,
                callee,
                MemoryLocation.Rdata);
            MergeReadOnlyDataAllocations(
                _destinationFunction.SchedResult.ChipLocalRdatas,
                callee.SchedResult.ChipLocalRdatas,
                _destinationFunction,
                callee,
                MemoryLocation.ChipLocalRdata);
            MergeReadOnlyDataAllocations(
                _destinationFunction.SchedResult.BlockLocalRdatas,
                callee.SchedResult.BlockLocalRdatas,
                _destinationFunction,
                callee,
                MemoryLocation.BlockLocalRdata);
        }
    }

    private sealed class PrimFunctionBodyCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replacements;
        private readonly IReadOnlyDictionary<MemoryLocation, TIR.Buffer> _workspaceBases;

        public PrimFunctionBodyCloner(
            IReadOnlyDictionary<BaseExpr, BaseExpr> replacements,
            IReadOnlyDictionary<MemoryLocation, TIR.Buffer> workspaceBases)
        {
            _replacements = replacements;
            _workspaceBases = workspaceBases;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (_replacements.TryGetValue(expr, out var replacement))
            {
                return replacement;
            }

            return base.DispatchVisit(expr, context);
        }

        protected override BaseExpr VisitLeafBuffer(TIR.Buffer expr, Unit context)
        {
            if (TryGetActualTensorBuffer(expr, out var actualBuffer) && IsDirectAbiDescriptor(expr))
            {
                return actualBuffer;
            }

            return base.VisitLeafBuffer(expr, context);
        }

        protected override BaseExpr VisitLeafPhysicalBuffer(PhysicalBuffer expr, Unit context)
        {
            if (!_workspaceBases.TryGetValue(expr.Location, out var workspaceBase))
            {
                return base.VisitLeafPhysicalBuffer(expr, context);
            }

            var localStart = Clone(expr.Start, context);
            var workspacePhysicalStart = workspaceBase.MemSpan.Buffer.Start;
            if (!TryGetFixedInt64(localStart, out var localStartBytes)
                || !TryGetFixedInt64(workspacePhysicalStart, out var workspaceStartBytes)
                || !workspaceBase.MemSpan.Start.IsFixed)
            {
                throw new InvalidOperationException(
                    $"Cannot inline bufferized workspace at {expr.Location}: local and caller workspace starts must be allocated, " +
                    "and the caller workspace view offset must be fixed.");
            }

            var rebasedStart = checked(
                workspaceStartBytes
                + workspaceBase.MemSpan.Start.FixedValue
                + localStartBytes);
            return expr.With(
                start: rebasedStart,
                size: Clone(expr.Size, context));
        }

        protected override BaseExpr VisitLeafMemSpan(MemSpan expr, Unit context)
        {
            var start = Clone(expr.Start, context);
            var size = Clone(expr.Size, context);
            if (expr.Buffer.Start is BaseExpr formalStorage
                && _replacements.TryGetValue(formalStorage, out var replacement)
                && replacement is TIR.Buffer actualBuffer)
            {
                var composedStart = (actualBuffer.MemSpan.Start + start).Simplify();
                return expr.With(
                    buffer: actualBuffer.MemSpan.Buffer,
                    start: composedStart,
                    size: size);
            }

            return expr.With(
                buffer: Clone(expr.Buffer, context),
                start: start,
                size: size);
        }

        private static bool IsDirectAbiDescriptor(TIR.Buffer buffer)
            => buffer.MemSpan.Start.Simplify() is Dimension start
                && start.IsFixed
                && start.FixedValue == 0;

        private static bool TryGetFixedInt64(BaseExpr expression, out long value)
        {
            switch (expression)
            {
                case None:
                    value = 0;
                    return true;
                case DimConst dimConst:
                    value = dimConst.Value;
                    return true;
                case Dimension { IsFixed: true } dimension:
                    value = dimension.FixedValue;
                    return true;
                case TensorConst { Value.Shape.IsScalar: true } tensorConst:
                    return TryReadScalarInt64(tensorConst.Value, out value);
                default:
                    value = 0;
                    return false;
            }
        }

        private static bool TryReadScalarInt64(Tensor tensor, out long value)
        {
            var scalar = tensor[Array.Empty<long>()];
            switch (scalar)
            {
                case sbyte signed8:
                    value = signed8;
                    return true;
                case byte unsigned8:
                    value = unsigned8;
                    return true;
                case short signed16:
                    value = signed16;
                    return true;
                case ushort unsigned16:
                    value = unsigned16;
                    return true;
                case int signed32:
                    value = signed32;
                    return true;
                case uint unsigned32:
                    value = unsigned32;
                    return true;
                case long signed64:
                    value = signed64;
                    return true;
                case ulong unsigned64 when unsigned64 <= long.MaxValue:
                    value = (long)unsigned64;
                    return true;
            }

            if (scalar is not null)
            {
                var scalarType = scalar.GetType();
                if (scalarType.IsGenericType
                    && scalarType.GetGenericTypeDefinition() == typeof(Pointer<>)
                    && scalarType.GetProperty(nameof(Pointer<byte>.Value))?.GetValue(scalar) is ulong pointer
                    && pointer <= long.MaxValue)
                {
                    value = (long)pointer;
                    return true;
                }
            }

            value = 0;
            return false;
        }

        private static bool UsesBackingTensorLogicalLayout(TIR.Buffer buffer, IVar backingParameter)
        {
            var backingTensorType = backingParameter.CheckedType switch
            {
                TensorType tensorType => tensorType,
                DistributedType distributedType => distributedType.TensorType,
                _ => null,
            };
            if (backingTensorType?.Shape is not RankedShape backingShape
                || buffer.Rank != backingShape.Rank
                || !buffer.ElemType.Equals(backingTensorType.DType)
                || !buffer.Dimensions.SequenceEqual(backingShape.Dimensions))
            {
                return false;
            }

            return Equals(buffer.DistributedType, backingParameter.CheckedType as DistributedType);
        }

        private bool TryGetActualTensorBuffer(TIR.Buffer buffer, out TIR.Buffer actualBuffer)
        {
            actualBuffer = null!;
            if (buffer.MemSpan.Buffer.Start is not IVar formalParameter
                || !_replacements.TryGetValue((BaseExpr)formalParameter, out var replacement)
                || replacement is not TIR.Buffer candidate
                || !UsesBackingTensorLogicalLayout(buffer, formalParameter))
            {
                return false;
            }

            actualBuffer = candidate;
            return true;
        }
    }
}
