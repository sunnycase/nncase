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
            var callers = func.Users.Where(x => x is Call or FunctionWrapper).ToArray();
            var isEntry = callers.Length == 0;
            var visitor = new TIRSelectionVisitor(this);
            (var newBody, var outBuffers) = visitor.Select(func);
            var inBuffers = func.Parameters.ToArray();

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
                var primWrapper = new PrimFunctionWrapper(input.Name, primFunc, inBuffers.Length);
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
        var outputBufferShapes = outputBuffers.Select(x => (ElemType: x.CheckedDataType, Shape: x.CheckedShape)).ToArray();
        foreach (var caller in callers)
        {
            var outputAllocs = outputBufferShapes.Select(x => IR.F.Buffer.Uninitialized(x.ElemType, TIR.MemoryLocation.Data, x.Shape));
            var newArgs = caller.Arguments.ToArray().Concat(outputAllocs).ToArray();
            var newCaller = caller.With(arguments: newArgs);
            ReplaceUtility.ReplaceAllUsesWith(caller, newCaller);
        }
    }

    private sealed record SelectionResult(Sequential Body, IReadOnlyList<BufferVar> OutputBuffers);

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

            return new(new Sequential(_body.ToArray()), outBuffers.Cast<BufferVar>().ToArray());
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

        private static BaseExpr[] FlattenOutputArguments(BaseExpr output)
            => output is IR.Tuple tuple ? tuple.Fields.ToArray() : new[] { output };

        protected override BaseExpr VisitLeafIf(If expr, Unit context)
        {
            var output = CreateOutputBuffer(expr);
            var condition = (Expr)Visit(expr.Condition, context);
            return T.Let(out var outputVar, (Expr)output).Body(
                T.Assign(out var arguments, expr.Arguments.AsValueEnumerable().Select(x => ExprMemo[x]).ToArray().Concat(output is IR.Tuple tupleOutput ? tupleOutput.Fields.ToArray() : new[] { outputVar }).ToArray()),
                T.If(condition)
                    .Then(new Call(new FunctionWrapper(_selectionPass.ModuleKind, expr.Then), arguments))
                    .Else(new Call(new FunctionWrapper(_selectionPass.ModuleKind, expr.Else), arguments)))
                .Build();
        }

        private BaseExpr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments)
        {
            if (call.Target is IR.Tensors.GetItem && arguments[IR.Tensors.GetItem.Input.Index] is IR.Tuple tuple && call[IR.Tensors.GetItem.Index] is DimConst index)
            {
                return tuple[index.Value];
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
