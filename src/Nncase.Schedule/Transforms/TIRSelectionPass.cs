// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule.TileGraph;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

public abstract class TIRSelectionPass : FunctionPass
{
    public TIRSelectionPass(string moduleKind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    /// <inheritdoc/>
    public override FunctionPassTraversalOrder TraversalOrder => FunctionPassTraversalOrder.CalleeFirst;

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is Function func)
        {
            var callers = func.Users.Where(x => x is Call or If or FunctionWrapper).ToArray();
            var isEntry = callers.Length == 0;
            var visitor = new TIRSelectionVisitor(this, isEntry);
            var selection = visitor.Select(func);
            var parameters = selection.InputBuffers.Concat(selection.OutputParameters).ToArray();

            if (isEntry)
            {
                var primFunc = new PrimFunction(
                    $"{input.Name}_prim",
                    ModuleKind,
                    selection.Body,
                    new Return(selection.Results.ToArray()),
                    parameters)
                {
                    Role = input.Role,
                };
                _ = primFunc.GetAbiView();
                return Task.FromResult((BaseFunction)primFunc);
            }
            else
            {
                // Allocate output buffers in the caller functions
                var primFunc = new PrimFunction(
                    $"{input.Name}_prim",
                    ModuleKind,
                    selection.Body,
                    new Return(selection.Results.ToArray()),
                    parameters)
                {
                    Role = input.Role,
                };
                var primWrapper = new PrimFunctionWrapper(input.Name, primFunc, selection.InputBuffers.Count);
                RewriteCallersForPrimFunction(primFunc, callers.OfType<Call>());
                return Task.FromResult((BaseFunction)primWrapper);
            }
        }

        return Task.FromResult(input);
    }

    protected abstract Expr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments, ref Expr output, TIRSelectionContext context);

    /// <summary>
    /// Finalizes a target-selected expression after its semantic TIR operands
    /// have been materialized.
    /// </summary>
    protected virtual Expr FinalizeSelectedExpr(
        Call sourceCall,
        Expr selectedExpr,
        TIRSelectionContext context)
        => selectedExpr;

    /// <summary>
    /// Applies target-owned physical storage requirements before binding a
    /// caller-allocated PrimFunction invocation.
    /// </summary>
    protected virtual IReadOnlyList<BaseExpr> PrepareCallerAllocatedPrimFunctionArguments(
        PrimFunction primFunction,
        IReadOnlyList<BaseExpr> arguments,
        TIRSelectionContext context)
        => arguments;

    /// <summary>
    /// Creates storage for an intermediate tensor result. Targets may override
    /// this when a distributed value requires a physical representation that is
    /// not a per-owner slice of the default data arena.
    /// </summary>
    protected virtual TIR.Buffer CreateIntermediateBuffer(
        Expr sourceExpr,
        IRType type,
        MemoryLocation defaultLocation,
        IReadOnlyList<int> resultPath,
        string name)
    {
        var tensorType = type switch
        {
            DistributedType distributedType => distributedType.TensorType,
            TensorType value => value,
            _ => throw new ArgumentException($"Unsupported intermediate buffer type: {type}"),
        };
        return T.CreateBuffer(
            tensorType,
            defaultLocation,
            out _,
            name,
            type as DistributedType);
    }

    /// <summary>
    /// Selects the physical tensor layout carried by a function ABI parameter.
    /// </summary>
    protected virtual BufferLayoutAnnotation? GetBufferParameterLayout(
        IRType type,
        BufferVarRole role,
        bool isEntry)
        => isEntry && IsRuntimeTensorType(type)
            ? BufferLayoutAnnotation.RuntimeStrided
            : null;

    /// <summary>
    /// Creates the caller-owned storage bound to a PrimFunction output
    /// parameter. The output ABI is authoritative for representations that
    /// require more than one physical owner component.
    /// </summary>
    protected virtual Expr CreateCallerAllocatedPrimFunctionOutput(
        TIR.PrimFunction primFunction,
        BufferVar outputParameter)
    {
        if (outputParameter.LayoutAnnotation.DistributedStorageKind ==
                DistributedBufferStorageKind.CompactPerOwner &&
            outputParameter.CheckedType is DistributedType distributedType)
        {
            return T.CreateCompactPerOwnerBuffer(
                distributedType,
                out _,
                $"{primFunction.Name}_{outputParameter.Name}_storage");
        }

        return outputParameter.CheckedType switch
        {
            DistributedType dt => IR.F.Buffer.Uninitialized(
                dt.TensorType.DType,
                TIR.MemoryLocation.Data,
                dt.TensorType.Shape,
                dt.AxisPolicies,
                dt.Placement,
                dt.Partial),
            TensorType tt => IR.F.Buffer.Uninitialized(
                tt.DType,
                TIR.MemoryLocation.Data,
                tt.Shape),
            var type => throw new InvalidOperationException(
                $"TIR selection caller-allocated output expects tensor type, got {type.GetType().Name}."),
        };
    }

    protected IRType GetArgumentType(BaseExpr argument)
    {
        return argument switch
        {
            TIR.Buffer b => b.DistributedType ?? b.CheckedType,
            _ => argument.CheckedType,
        };
    }

    private static bool IsRuntimeTensorType(IRType type)
        => type switch
        {
            TensorType { DType: not (PointerType or ReferenceType), Shape: RankedShape } => true,
            DistributedType { TensorType: { DType: not (PointerType or ReferenceType), Shape: RankedShape } } => true,
            _ => false,
        };

    private void RewriteCallersForPrimFunction(
        PrimFunction primFunction,
        IEnumerable<Call> callers)
    {
        var outputParameters = primFunction.GetAbiView().OutputParameters.ToArray();
        foreach (var caller in callers)
        {
            var outputAllocs = outputParameters
                .Select(outputParameter =>
                    CreateCallerAllocatedPrimFunctionOutput(primFunction, outputParameter))
                .ToArray();
            var newArgs = caller.Arguments.ToArray().Concat(outputAllocs).ToArray();
            var newCaller = caller.With(arguments: newArgs);
            ReplaceUtility.ReplaceAllUsesWith(caller, newCaller);
        }
    }

    private sealed record SelectionResult(
        Sequential Body,
        IReadOnlyList<IVar> InputBuffers,
        IReadOnlyList<BufferVar> OutputParameters,
        IReadOnlyList<Expr> Results);

    private sealed class TIRSelectionVisitor : ExprCloner<Unit>
    {
        private readonly TIRSelectionPass _selectionPass;
        private readonly TIRSelectionContext _selectionContext;
        private readonly bool _isEntry;
        private readonly List<Expr> _body = new();
        private int _bufferIndex;

        public TIRSelectionVisitor(TIRSelectionPass selectionPass, bool isEntry)
        {
            _selectionPass = selectionPass;
            _selectionContext = new TIRSelectionContext(ExprMemo);
            _isEntry = isEntry;
        }

        public SelectionResult Select(Function function)
        {
            var inBuffers = LowerInputParameters(function.Parameters).ToArray();
            Visit(function.Body, Unit.Default);

            var results = function.Body switch
            {
                IR.Tuple tuple => tuple.Fields.AsValueEnumerable().Select(x => (Expr)ExprMemo[x]).ToArray(),
                var body => ExprMemo[function.Body] switch
                {
                    IR.Tuple bodyTuple => bodyTuple.Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(),
                    var x => [(Expr)x],
                },
            };

            var outputParameters = new List<BufferVar>();
            var promotedStorages = new Dictionary<PhysicalBuffer, BufferVar>(ReferenceEqualityComparer.Instance);
            for (var resultIndex = 0; resultIndex < results.Length; resultIndex++)
            {
                switch (results[resultIndex])
                {
                    case BufferVar { Role: BufferVarRole.Output } outputParameter:
                        AddOutputParameter(outputParameter);
                        break;
                    case BufferVar { Role: BufferVarRole.Input or BufferVarRole.InOut }:
                        break;
                    case TIR.Buffer buffer:
                        PromoteResultStorage(buffer, resultIndex);
                        break;
                    default:
                        throw new InvalidOperationException(
                            $"Function {function.Name} result {resultIndex} must lower to BufferVar or TIR.Buffer, got {results[resultIndex].GetType().Name}.");
                }
            }

            return new(new Sequential(_body.ToArray()), inBuffers, outputParameters, results);

            void AddOutputParameter(BufferVar outputParameter)
            {
                if (!outputParameters.Contains(outputParameter, ReferenceEqualityComparer.Instance))
                {
                    outputParameters.Add(outputParameter);
                }
            }

            void PromoteResultStorage(TIR.Buffer buffer, int resultIndex)
            {
                var physicalBuffer = buffer.MemSpan.Buffer;
                switch (physicalBuffer.Start)
                {
                    case BufferVar { Role: BufferVarRole.Output } outputParameter:
                        AddOutputParameter(outputParameter);
                        return;
                    case BufferVar { Role: BufferVarRole.Input or BufferVarRole.InOut }:
                        return;
                    case None when physicalBuffer.Location is MemoryLocation.Data or
                        MemoryLocation.Cache or
                        MemoryLocation.ChipLocalData:
                        if (!CanPromoteCompleteResultStorage(buffer))
                        {
                            throw new InvalidOperationException(
                                $"Function {function.Name} result {resultIndex} is a partial view of internal storage. " +
                                "Insert an explicit materialization before promoting it to caller-allocated output storage.");
                        }

                        if (!promotedStorages.TryGetValue(physicalBuffer, out var promotedOutput))
                        {
                            var promotedLayout = _isEntry
                                ? null
                                : BufferLayoutAnnotation.ExactStrided(
                                    buffer.Strides,
                                    buffer.DistributedStorageKind);
                            promotedOutput = CreateOutputBufferVar(
                                _selectionPass.GetArgumentType(buffer),
                                promotedLayout);
                            promotedStorages.Add(physicalBuffer, promotedOutput);
                            AddOutputParameter(promotedOutput);
                            var outputStorage = physicalBuffer.With(start: promotedOutput, location: MemoryLocation.Output);
                            ReplaceUtility.ReplaceAllUsesWith(physicalBuffer, outputStorage);
                        }

                        return;
                    default:
                        throw new InvalidOperationException(
                            $"Function {function.Name} result {resultIndex} is backed by non-ABI storage {physicalBuffer.Location}/{physicalBuffer.Start.GetType().Name}. Insert an explicit materialization before TIR selection.");
                }
            }

            static bool CanPromoteCompleteResultStorage(TIR.Buffer buffer)
            {
                if (!buffer.MemSpan.Start.Simplify().Equals(Dimension.Zero))
                {
                    return false;
                }

                var physicalSize = buffer.MemSpan.Buffer.Size.Simplify();
                var componentSize = buffer.MemSpan.Size.Simplify();
                if (buffer.DistributedStorageKind != DistributedBufferStorageKind.CompactPerOwner)
                {
                    return componentSize.Equals(physicalSize);
                }

                if (buffer.DistributedType is not { } distributedType ||
                    buffer.MemSpan.Buffer.Location != MemoryLocation.ChipLocalData)
                {
                    return false;
                }

                var ownerCount = distributedType.Placement.Hierarchy.Aggregate(
                    1L,
                    (product, extent) => checked(product * extent));
                return (componentSize * ownerCount).Simplify().Equals(physicalSize);
            }
        }

        protected sealed override Expr VisitLeafTensorConst(TensorConst expr, Unit context)
        {
            return T.AttachBuffer(expr, out _, $"const_{_bufferIndex++}");
        }

        protected sealed override Expr VisitLeafVar(Var expr, Unit context) => expr;

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (HasVisited(expr, out var result))
            {
                return result;
            }

            result = base.DispatchVisit(expr, context);
            _selectionContext.RegisterSelectedValue(expr, result);
            return result;
        }

        protected override BaseExpr VisitCall(Call expr, Unit context)
        {
            foreach (var argument in expr.Arguments)
            {
                Visit(argument, context);
            }

            return VisitLeafCall(expr, context);
        }

        protected override BaseExpr VisitGrid(Grid expr, Unit context)
        {
            var analysis = new GridMemoryEffectAnalysis().Analyze(expr);
            if (analysis.BufferAliases.Length == 0 ||
                analysis.Effects.Any(effect => effect.Mode != MemoryAccessMode.None))
            {
                throw new InvalidOperationException(
                    $"Executable Grid survived AutoTiling before TIR selection: {expr}. Only residual buffer aliases are legal here.");
            }

            var writeAccessIndices = expr.Accesses.ToArray()
                .Select((access, index) => (Access: access, Index: index))
                .Where(pair => pair.Access.IsWrite)
                .Select(pair => pair.Index)
                .ToArray();
            var results = new Expr[writeAccessIndices.Length];
            for (var outputIndex = 0; outputIndex < writeAccessIndices.Length; outputIndex++)
            {
                var resultAccessIndex = writeAccessIndices[outputIndex];
                var aliases = analysis.BufferAliases
                    .Where(alias => alias.ResultAccessIndex == resultAccessIndex)
                    .ToArray();
                if (aliases.Length != 1)
                {
                    throw new InvalidOperationException(
                        $"Residual buffer alias Grid output access {resultAccessIndex} must have exactly one storage source, got {aliases.Length}.");
                }

                results[outputIndex] = LowerResidualBufferAlias(
                    expr,
                    aliases[0],
                    context,
                    outputIndex);
            }

            return results.Length switch
            {
                0 => throw new InvalidOperationException("Residual buffer alias Grid must have at least one output."),
                1 => results[0],
                _ => new IR.Tuple(results),
            };
        }

        protected sealed override BaseExpr VisitLeafCall(Call expr, Unit context)
        {
            var args = expr.Arguments.AsValueEnumerable().Select(x => ExprMemo[x]).ToArray();
            return SelectCall(expr, args);
        }

        protected override BaseExpr VisitIf(If expr, Unit context)
        {
            if (ConditionUsesTensorValue(expr.Condition))
            {
                Visit(expr.Condition, context);
            }

            foreach (var argument in expr.Arguments)
            {
                Visit(argument, context);
            }

            return VisitLeafIf(expr, context);
        }

        protected override BaseExpr VisitLeafIf(If expr, Unit context)
        {
            var output = CreateOutputBuffer(expr);
            var condition = ConditionUsesTensorValue(expr.Condition)
                ? (Expr)Visit(expr.Condition, context)
                : expr.Condition;
            var arguments = expr.Arguments.AsValueEnumerable().Select(x => ExprMemo[x]).ToArray().Concat(FlattenOutputArguments(output)).ToArray();
            _body.Add(T.If(condition)
                .Then(new Call(WrapIfBranch(expr.Then), arguments))
                .Else(new Call(WrapIfBranch(expr.Else), arguments))
                .Build());
            return output;
        }

        private TIR.Buffer LowerResidualBufferAlias(
            Grid grid,
            GridBufferAlias alias,
            Unit context,
            int outputIndex)
        {
            var sourceAccess = grid.Accesses[alias.SourceAccessIndex];
            var resultAccess = grid.Accesses[alias.ResultAccessIndex];
            var selectedSource = Visit(sourceAccess.Value, context);
            var sourceBuffer = selectedSource switch
            {
                TIR.Buffer buffer => buffer,
                BufferVar bufferVar => AttachBufferVar(bufferVar, sourceAccess.Value.CheckedType, grid, outputIndex),
                _ => throw new InvalidOperationException(
                    $"Residual buffer alias Grid source must lower to TIR.Buffer, got {selectedSource.GetType().Name}."),
            };

            var transform = new BufferViewTransform(
                sourceAccess.AffineMap,
                resultAccess.AffineMap,
                new IRArray<Dimension>(grid.DomainBounds.ToArray()));
            return BufferViewUtility.CreateLogicalBufferView(
                sourceBuffer,
                resultAccess.Value.CheckedType,
                transform,
                $"alias_grid_{_bufferIndex++}_out{outputIndex}");

            static TIR.Buffer AttachBufferVar(BufferVar bufferVar, IRType sourceType, Grid grid, int outputIndex)
            {
                if (!TryGetTensorType(sourceType, out var tensorType, out var distributedType))
                {
                    throw new InvalidOperationException(
                        $"Residual buffer alias Grid source {grid} output {outputIndex} must have a tensor type, got {sourceType}.");
                }

                return T.AttachBuffer(
                    bufferVar,
                    tensorType,
                    bufferVar.Location,
                    0,
                    out _,
                    $"{bufferVar.Name}_alias_source",
                    distributedType);
            }
        }

        private BaseFunction WrapIfBranch(BaseFunction branch)
            => branch switch
            {
                FunctionWrapper { Target: PrimFunctionWrapper primFunctionWrapper } => primFunctionWrapper.Target,
                FunctionWrapper wrapper => wrapper,
                TIR.PrimFunction primFunction => primFunction,
                PrimFunctionWrapper primFunctionWrapper => primFunctionWrapper.Target,
                Function => new FunctionWrapper(_selectionPass.ModuleKind, branch),
                _ => throw new InvalidOperationException($"TIR selection if branch expects Function, FunctionWrapper, PrimFunctionWrapper or PrimFunction, got {branch.GetType().Name}."),
            };

        private static bool ConditionUsesTensorValue(Expr condition)
            => ExprCollector.Collect(condition).Any(expr =>
                (expr is Var or TensorConst or TIR.Buffer)
                && expr.CheckedType is TensorType or DistributedType);

        private static BaseExpr[] FlattenOutputArguments(BaseExpr output)
            => output is IR.Tuple tuple ? tuple.Fields.ToArray() : new[] { output };

        private static bool TryGetTensorType(IRType type, out TensorType tensorType, out DistributedType? distributedType)
        {
            switch (type)
            {
                case DistributedType dt:
                    tensorType = dt.TensorType;
                    distributedType = dt;
                    return true;
                case TensorType tt when tt.DType is not PointerType:
                    tensorType = tt;
                    distributedType = null;
                    return true;
                default:
                    tensorType = null!;
                    distributedType = null;
                    return false;
            }
        }

        private bool TrySelectCallerAllocatedPrimCall(
            Expr target,
            IReadOnlyList<BaseExpr> arguments,
            TIRSelectionContext context,
            out Call selectedCall,
            out BaseExpr output)
        {
            selectedCall = null!;
            output = null!;
            if (!TryGetPrimFunctionTarget(target, out var primFunction))
            {
                return false;
            }

            var abi = primFunction.GetAbiView();
            if (abi.Results.Count == 0)
            {
                return false;
            }

            var parameters = primFunction.Parameters.ToArray();
            foreach (var outputParameter in abi.OutputParameters)
            {
                var parameterIndex = Array.FindIndex(parameters, parameter => ReferenceEquals(parameter, outputParameter));
                if (parameterIndex < 0)
                {
                    throw new InvalidOperationException($"PrimFunction {primFunction.Name} ABI output {outputParameter.Name} is not in its parameter list.");
                }

                if (parameterIndex >= arguments.Count)
                {
                    return false;
                }
            }

            var preparedArguments = _selectionPass.PrepareCallerAllocatedPrimFunctionArguments(
                primFunction,
                arguments,
                context);
            if (preparedArguments.Count != arguments.Count)
            {
                throw new InvalidOperationException(
                    $"PrimFunction {primFunction.Name} argument preparation changed arity from " +
                    $"{arguments.Count} to {preparedArguments.Count}.");
            }

            var normalizedArguments = NormalizePrimFunctionArguments(primFunction, preparedArguments);
            var resultValues = abi.Results.Select((result, resultIndex) =>
            {
                var storageIndex = Array.FindIndex(parameters, parameter => ReferenceEquals(parameter, result.Storage));
                if (storageIndex < 0 || storageIndex >= preparedArguments.Count)
                {
                    throw new InvalidOperationException(
                        $"PrimFunction {primFunction.Name} result {resultIndex} storage {result.Storage.Name} is not bound by the call arguments.");
                }

                // ABI normalization is local to the callee invocation. Keep the caller-visible
                // result attached to the prepared logical storage supplied by the caller.
                var storage = preparedArguments[storageIndex];
                return result.Value switch
                {
                    BufferVar => storage,
                    TIR.Buffer resultView => CreateResultView(resultView, storage, primFunction.Name, resultIndex),
                    _ => throw new InvalidOperationException(
                        $"PrimFunction {primFunction.Name} result {resultIndex} has unsupported binding {result.Value.GetType().Name}."),
                };
            }).ToArray();
            output = resultValues.Length == 1 ? (Expr)resultValues[0] : new IR.Tuple(resultValues.Cast<Expr>().ToArray());
            selectedCall = new Call(primFunction, normalizedArguments);
            return true;

            static BaseExpr CreateResultView(TIR.Buffer descriptor, BaseExpr storage, string functionName, int resultIndex)
            {
                if (storage is not TIR.Buffer storageBuffer)
                {
                    throw new InvalidOperationException(
                        $"PrimFunction {functionName} result {resultIndex} view storage must lower to TIR.Buffer, got {storage.GetType().Name}.");
                }

                return T.CreateBufferView(
                    storageBuffer,
                    descriptor.ElemType,
                    descriptor.Dimensions.ToArray(),
                    descriptor.Strides.ToArray(),
                    descriptor.MemSpan.Start,
                    descriptor.MemSpan.Size,
                    descriptor.DistributedType,
                    $"{functionName}_result_{resultIndex}");
            }
        }

        private static Call CreatePrimFunctionCall(TIR.PrimFunction primFunction, IReadOnlyList<BaseExpr> arguments)
            => new(primFunction, NormalizePrimFunctionArguments(primFunction, arguments));

        private static BaseExpr[] NormalizePrimFunctionArguments(TIR.PrimFunction primFunction, IReadOnlyList<BaseExpr> arguments)
        {
            var parameters = primFunction.Parameters.ToArray();
            if (parameters.Length != arguments.Count)
            {
                throw new InvalidOperationException(
                    $"PrimFunction {primFunction.Name} expects {parameters.Length} arguments, got {arguments.Count}.");
            }

            var normalized = arguments.ToArray();
            for (var index = 0; index < parameters.Length; index++)
            {
                var parameter = parameters[index];
                if (!TryGetTensorType(parameter.CheckedType, out _, out _))
                {
                    continue;
                }

                if (normalized[index] is not TIR.Buffer argumentBuffer)
                {
                    if (!Equals(normalized[index].CheckedType, parameter.CheckedType))
                    {
                        throw new InvalidOperationException(
                            $"PrimFunction {primFunction.Name} tensor parameter {parameter.Name} expects {parameter.CheckedType}, " +
                            $"but argument {index} is {normalized[index].GetType().Name} with type {normalized[index].CheckedType}.");
                    }

                    continue;
                }

                var argumentType = (IRType?)argumentBuffer.DistributedType ??
                    new TensorType(argumentBuffer.ElemType, new RankedShape(argumentBuffer.Dimensions.ToArray()));
                if (Equals(argumentType, parameter.CheckedType))
                {
                    continue;
                }

                if (!BufferViewUtility.TryCreate(argumentType, parameter.CheckedType, out var transform))
                {
                    throw new InvalidOperationException(
                        $"PrimFunction {primFunction.Name} tensor parameter {parameter.Name} expects {parameter.CheckedType}, " +
                        $"but argument {index} has storage-incompatible type {argumentType}.");
                }

                normalized[index] = BufferViewUtility.CreateLogicalBufferView(
                    argumentBuffer,
                    parameter.CheckedType,
                    transform,
                    $"{primFunction.Name}_arg{index}_abi_view");
            }

            return normalized;
        }

        private static bool TryGetPrimFunctionTarget(Expr target, out TIR.PrimFunction primFunction)
        {
            switch (target)
            {
                case TIR.PrimFunction direct:
                    primFunction = direct;
                    return true;
                case PrimFunctionWrapper wrapper:
                    primFunction = wrapper.Target;
                    return true;
                case FunctionWrapper { Target: PrimFunctionWrapper wrapper }:
                    primFunction = wrapper.Target;
                    return true;
                default:
                    primFunction = null!;
                    return false;
            }
        }

        private IReadOnlyList<IVar> LowerInputParameters(ReadOnlySpan<IVar> parameters)
        {
            var lowered = new IVar[parameters.Length];
            for (var i = 0; i < parameters.Length; i++)
            {
                var parameter = parameters[i];
                if (TryGetTensorType(parameter.CheckedType, out var tensorType, out var distributedType))
                {
                    var bufferVar = parameter is BufferVar { Role: BufferVarRole.Input or BufferVarRole.InOut } inputBufferVar
                        ? inputBufferVar
                        : new BufferVar(
                            parameter.Name,
                            parameter.CheckedType,
                            BufferVarRole.Input,
                            MemoryLocation.Input,
                            _selectionPass.GetBufferParameterLayout(
                                parameter.CheckedType,
                                BufferVarRole.Input,
                                _isEntry));
                    var buffer = T.AttachBuffer(bufferVar, tensorType, MemoryLocation.Input, 0, out _, $"{parameter.Name}_input", distributedType);
                    ExprMemo[(BaseExpr)parameter] = buffer;
                    _selectionContext.RegisterSelectedValue((BaseExpr)parameter, buffer);
                    lowered[i] = bufferVar;
                }
                else
                {
                    ExprMemo[(BaseExpr)parameter] = (BaseExpr)parameter;
                    _selectionContext.RegisterSelectedValue((BaseExpr)parameter, (BaseExpr)parameter);
                    lowered[i] = parameter;
                }
            }

            return lowered;
        }

        private BaseExpr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments)
        {
            if (call.Target is IR.Tensors.GetItem && arguments[IR.Tensors.GetItem.Input.Index] is IR.Tuple tuple && call[IR.Tensors.GetItem.Index] is DimConst index)
            {
                return tuple[index.Value];
            }
            else if (TrySelectCallerAllocatedPrimCall(
                         call.Target,
                         arguments,
                         _selectionContext,
                         out var callerAllocatedCall,
                         out var callerAllocatedOutput))
            {
                _body.Add(callerAllocatedCall);
                return callerAllocatedOutput;
            }
            else
            {
                var output = CreateOutputBuffer(call);
                var newCall = call.Target switch
                {
                    TIR.PrimFunction deviceFunc => CreatePrimFunctionCall(deviceFunc, arguments.Concat(FlattenOutputArguments(output)).ToArray()),
                    PrimFunctionWrapper { Target: TIR.PrimFunction deviceFunc } => CreatePrimFunctionCall(deviceFunc, arguments.Concat(FlattenOutputArguments(output)).ToArray()),
                    Function fn => new Call(new FunctionWrapper(_selectionPass.ModuleKind, fn), arguments.Concat(FlattenOutputArguments(output)).ToArray()),
                    _ => _selectionPass.FinalizeSelectedExpr(
                        call,
                        _selectionPass.SelectCall(call, arguments, ref Unsafe.As<BaseExpr, Expr>(ref output), _selectionContext),
                        _selectionContext),
                };
                _body.Add(newCall);
                return output;
            }
        }

        private BaseExpr CreateOutputBuffer(Expr expr)
        {
            var root = VisitRoot!;
            var memoryLocation = MemoryLocation.Data;
            if (ReferenceEquals(root, expr)
                || (root is IR.Tuple tuple && tuple.Fields.AsValueEnumerable().Contains(expr, ReferenceEqualityComparer.Instance)))
            {
                return CreateOutputBuffer(expr.CheckedType);
            }

            if (expr.CheckedType is TupleType tt)
            {
                var fields = tt.Fields.AsValueEnumerable()
                    .Select((type, index) => IsRootTupleProjection(expr, index)
                        ? CreateOutputBuffer(type)
                        : CreateIntermediateOutput(expr, type, memoryLocation, [index]))
                    .ToArray();
                return new IR.Tuple(fields);
            }
            else
            {
                return CreateBuffer(expr, expr.CheckedType, memoryLocation);
            }
        }

        private bool IsRootTupleProjection(Expr tupleValue, int fieldIndex)
        {
            if (IsProjection(VisitRoot, tupleValue, fieldIndex))
            {
                return true;
            }

            return VisitRoot is IR.Tuple rootTuple &&
                rootTuple.Fields.AsValueEnumerable().Any(field =>
                    IsProjection(field, tupleValue, fieldIndex));

            static bool IsProjection(BaseExpr? value, Expr tupleValue, int fieldIndex)
                => value is Call projection &&
                projection.Target is IR.Tensors.GetItem &&
                ReferenceEquals(projection[IR.Tensors.GetItem.Input], tupleValue) &&
                projection[IR.Tensors.GetItem.Index] is DimConst index &&
                index.Value == fieldIndex;
        }

        private BaseExpr CreateIntermediateOutput(
            Expr sourceExpr,
            IRType type,
            MemoryLocation memoryLocation,
            IReadOnlyList<int> resultPath)
            => type is TupleType tupleType
                ? new IR.Tuple(tupleType.Fields.AsValueEnumerable()
                    .Select((field, index) => CreateIntermediateOutput(sourceExpr, field, memoryLocation, [.. resultPath, index]))
                    .ToArray())
                : CreateBuffer(sourceExpr, type, memoryLocation, resultPath);

        private BaseExpr CreateOutputBuffer(IRType type)
        {
            if (type is TupleType tupleType)
            {
                return new IR.Tuple(tupleType.Fields.AsValueEnumerable().Select(CreateOutputBuffer).ToArray());
            }

            var outputParameter = CreateOutputBufferVar(type);
            if (!TryGetTensorType(type, out var tensorType, out var distributedType))
            {
                return outputParameter;
            }

            return T.AttachBuffer(
                outputParameter,
                tensorType,
                MemoryLocation.Output,
                0,
                out _,
                $"{outputParameter.Name}_view",
                distributedType);
        }

        private BufferVar CreateOutputBufferVar(
            IRType type,
            BufferLayoutAnnotation? layoutAnnotation = null)
        {
            var layout = layoutAnnotation ??
                _selectionPass.GetBufferParameterLayout(type, BufferVarRole.Output, _isEntry);
            return new BufferVar(
                $"out_{_bufferIndex++}",
                type,
                BufferVarRole.Output,
                MemoryLocation.Output,
                layout);
        }

        private TIR.Buffer CreateBuffer(Expr sourceExpr, IRType type, MemoryLocation memoryLocation)
            => CreateBuffer(sourceExpr, type, memoryLocation, Array.Empty<int>());

        private TIR.Buffer CreateBuffer(
            Expr sourceExpr,
            IRType type,
            MemoryLocation memoryLocation,
            IReadOnlyList<int> resultPath)
            => _selectionPass.CreateIntermediateBuffer(
                sourceExpr,
                type,
                memoryLocation,
                resultPath,
                $"buffer_{_bufferIndex++}");
    }
}

public sealed class TIRSelectionContext
{
    private readonly Dictionary<BaseExpr, BaseExpr> _exprMemo;
    private readonly Dictionary<BaseExpr, HashSet<BaseExpr>> _selectedValueUsers = new(ReferenceEqualityComparer.Instance);

    internal TIRSelectionContext(Dictionary<BaseExpr, BaseExpr> exprMemo)
    {
        _exprMemo = exprMemo;
    }

    public void RegisterSelectedValue(BaseExpr source, BaseExpr selectedValue)
    {
        if (!_selectedValueUsers.TryGetValue(selectedValue, out var users))
        {
            users = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            _selectedValueUsers.Add(selectedValue, users);
        }

        users.Add(source);
    }

    public void ReplaceSelectedValue(BaseExpr oldValue, BaseExpr newValue)
    {
        if (!_selectedValueUsers.Remove(oldValue, out var users))
        {
            return;
        }

        if (!_selectedValueUsers.TryGetValue(newValue, out var newUsers))
        {
            newUsers = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            _selectedValueUsers.Add(newValue, newUsers);
        }

        foreach (var user in users)
        {
            if (_exprMemo.TryGetValue(user, out var selectedValue) && ReferenceEquals(selectedValue, oldValue))
            {
                _exprMemo[user] = newValue;
                newUsers.Add(user);
            }
        }
    }
}
