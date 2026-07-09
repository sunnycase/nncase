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

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is Function func)
        {
            var callers = func.Users.Where(x => x is Call or If or FunctionWrapper).ToArray();
            var isEntry = callers.Length == 0;
            var visitor = new TIRSelectionVisitor(this);
            (var newBody, var inBuffers, var outBuffers) = visitor.Select(func);

            if (isEntry)
            {
                var primFunc = new PrimFunction(
                    $"{input.Name}_prim",
                    ModuleKind,
                    newBody,
                    inBuffers.Concat(outBuffers).ToArray());
                return Task.FromResult((BaseFunction)primFunc);
            }
            else
            {
                // Allocate output buffers in the caller functions
                var primFunc = new PrimFunction(
                    $"{input.Name}_prim",
                    ModuleKind,
                    newBody,
                    inBuffers.Concat(outBuffers).ToArray());
                var primWrapper = new PrimFunctionWrapper(input.Name, primFunc, inBuffers.Count);
                AddOutputBufferAllocsToCallers(func, outBuffers, callers.OfType<Call>());
                return Task.FromResult((BaseFunction)primWrapper);
            }
        }

        return Task.FromResult(input);
    }

    protected abstract Expr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments, ref Expr output, TIRSelectionContext context);

    protected IRType GetArgumentType(BaseExpr argument)
    {
        return argument switch
        {
            TIR.Buffer b => b.DistributedType ?? b.CheckedType,
            _ => argument.CheckedType,
        };
    }

    private void AddOutputBufferAllocsToCallers(Function function, IEnumerable<BufferVar> outputBuffers, IEnumerable<Call> callers)
    {
        var outputBufferTypes = outputBuffers.Select(x => x.CheckedType).ToArray();
        foreach (var caller in callers)
        {
            var outputAllocs = outputBufferTypes.Select(CreateDataUninitialized);
            var newArgs = caller.Arguments.ToArray().Concat(outputAllocs).ToArray();
            var newCaller = caller.With(arguments: newArgs);
            ReplaceUtility.ReplaceAllUsesWith(caller, newCaller);
        }

        static Expr CreateDataUninitialized(IRType type)
            => type switch
            {
                DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape, dt.AxisPolicies, dt.Placement),
                TensorType tt => IR.F.Buffer.Uninitialized(tt.DType, TIR.MemoryLocation.Data, tt.Shape),
                _ => throw new InvalidOperationException($"TIR selection caller-allocated output expects tensor type, got {type.GetType().Name}."),
            };
    }

    private sealed record SelectionResult(Sequential Body, IReadOnlyList<IVar> InputBuffers, IReadOnlyList<BufferVar> OutputBuffers);

    private sealed class TIRSelectionVisitor : ExprCloner<Unit>
    {
        private readonly TIRSelectionPass _selectionPass;
        private readonly TIRSelectionContext _selectionContext;
        private readonly List<Expr> _body = new();
        private int _bufferIndex;

        public TIRSelectionVisitor(TIRSelectionPass selectionPass)
        {
            _selectionPass = selectionPass;
            _selectionContext = new TIRSelectionContext(ExprMemo);
        }

        public SelectionResult Select(Function function)
        {
            var inBuffers = LowerInputParameters(function.Parameters);
            Visit(function.Body, Unit.Default);

            var outBuffers = function.Body switch
            {
                IR.Tuple tuple => tuple.Fields.AsValueEnumerable().Select(x => (Expr)ExprMemo[x]).ToArray(),
                var body => ExprMemo[function.Body] switch
                {
                    IR.Tuple bodyTuple => bodyTuple.Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(),
                    var x => [(Expr)x],
                },
            };

            // Add necessary copy calls
            for (int i = 0; i < outBuffers.Length; i++)
            {
                var previousBuffers = outBuffers.AsReadOnlySpan(0, i);
                var currentBuffer = outBuffers[i];
                if (currentBuffer is not BufferVar { Role: BufferVarRole.Output }
                    || previousBuffers.ReferenceContains(currentBuffer)
                    || (currentBuffer is IVar currentVar && function.Parameters.ReferenceContains(currentVar)))
                {
                    var newBuffer = CreateOutputBufferVar(_selectionPass.GetArgumentType(currentBuffer));
                    _body.Add(T.Memcopy(newBuffer, currentBuffer));
                    outBuffers[i] = newBuffer;
                }
            }

            return new(new Sequential(_body.ToArray()), inBuffers, outBuffers.Cast<BufferVar>().ToArray());
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

        private static bool TrySelectCallerAllocatedPrimCall(Expr target, IReadOnlyList<BaseExpr> arguments, out Call selectedCall, out BaseExpr output)
        {
            selectedCall = null!;
            output = null!;
            if (!TryGetPrimFunctionTarget(target, out var primFunction, out var inputCount))
            {
                return false;
            }

            var outputCount = primFunction.GetAbiView().Outputs.Count;
            if (outputCount == 0 || arguments.Count < inputCount + outputCount)
            {
                return false;
            }

            var outputArguments = arguments.Skip(inputCount).Take(outputCount).ToArray();
            output = outputArguments.Length == 1 ? outputArguments[0] : new IR.Tuple(outputArguments);
            selectedCall = new Call(primFunction, arguments.ToArray());
            return true;
        }

        private static bool TryGetPrimFunctionTarget(Expr target, out TIR.PrimFunction primFunction, out int inputCount)
        {
            switch (target)
            {
                case TIR.PrimFunction direct:
                    primFunction = direct;
                    inputCount = direct.GetAbiView().Inputs.Count;
                    return true;
                case PrimFunctionWrapper wrapper:
                    primFunction = wrapper.Target;
                    inputCount = wrapper.ParametersCount;
                    return true;
                case FunctionWrapper { Target: PrimFunctionWrapper wrapper }:
                    primFunction = wrapper.Target;
                    inputCount = wrapper.ParametersCount;
                    return true;
                default:
                    primFunction = null!;
                    inputCount = 0;
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
                        : new BufferVar(parameter.Name, parameter.CheckedType, BufferVarRole.Input, MemoryLocation.Input);
                    var buffer = T.AttachBuffer(bufferVar, tensorType, MemoryLocation.Input, 0, out _, $"{parameter.Name}_input", distributedType);
                    ExprMemo[(BaseExpr)parameter] = buffer;
                    _selectionContext.RegisterSelectedValue((BaseExpr)parameter, buffer);
                    lowered[i] = bufferVar;
                }
                else
                {
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
            else if (TrySelectCallerAllocatedPrimCall(call.Target, arguments, out var callerAllocatedCall, out var callerAllocatedOutput))
            {
                _body.Add(callerAllocatedCall);
                return callerAllocatedOutput;
            }
            else
            {
                var output = CreateOutputBuffer(call);
                var newCall = call.Target switch
                {
                    TIR.PrimFunction deviceFunc => new Call(deviceFunc, arguments.Concat(FlattenOutputArguments(output)).ToArray()),
                    PrimFunctionWrapper { Target: TIR.PrimFunction deviceFunc } => new Call(deviceFunc, arguments.Concat(FlattenOutputArguments(output)).ToArray()),
                    Function fn => new Call(new FunctionWrapper(_selectionPass.ModuleKind, fn), arguments.Concat(FlattenOutputArguments(output)).ToArray()),
                    _ => _selectionPass.SelectCall(call, arguments, ref Unsafe.As<BaseExpr, Expr>(ref output), _selectionContext),
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
                var fields = tt.Fields.AsValueEnumerable().Select(x => CreateBuffer(x, memoryLocation)).ToArray();
                return new IR.Tuple(fields);
            }
            else
            {
                return CreateBuffer(expr.CheckedType, memoryLocation);
            }
        }

        private BaseExpr CreateOutputBuffer(IRType type)
        {
            if (type is TupleType tupleType)
            {
                return new IR.Tuple(tupleType.Fields.AsValueEnumerable().Select(CreateOutputBuffer).ToArray());
            }

            return CreateOutputBufferVar(type);
        }

        private BufferVar CreateOutputBufferVar(IRType type)
            => new($"out_{_bufferIndex++}", type, BufferVarRole.Output, MemoryLocation.Output);

        private TIR.Buffer CreateBuffer(IRType type, MemoryLocation memoryLocation)
        {
            var tensorType = type switch
            {
                DistributedType dt => dt.TensorType,
                TensorType tt => tt,
                _ => throw new ArgumentException($"Unsupported type: {type}"),
            };
            return T.CreateBuffer(tensorType, memoryLocation, out _, $"buffer_{_bufferIndex++}", type as DistributedType);
        }
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
