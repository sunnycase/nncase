// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Specializes internal prim functions by the physical layouts of their actual buffer arguments.
/// </summary>
public sealed class SpecializePrimFunctionBufferLayoutsPass : ModulePass
{
    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        if (input.Entry is not PrimFunction entry)
        {
            throw new InvalidOperationException(
                $"{nameof(SpecializePrimFunctionBufferLayoutsPass)} expects a lowered PrimFunction entry, " +
                $"but found {input.Entry?.GetType().Name ?? "null"}.");
        }

        var specializer = new BufferLayoutSpecializer(input);
        return Task.FromResult(specializer.Specialize(entry));
    }

    private sealed class BufferLayoutSpecializer
    {
        private readonly IRModule _module;
        private readonly Dictionary<PrimFunction, Dictionary<FunctionLayoutSignature, FunctionVariant>> _variants = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<FunctionVariant> _active = new(ReferenceEqualityComparer.Instance);
        private readonly List<FunctionVariant> _postOrder = new();

        public BufferLayoutSpecializer(IRModule module)
        {
            _module = module;
        }

        public IRModule Specialize(PrimFunction entry)
        {
            var entryVariant = Discover(entry, GetDeclaredSignature(entry));
            AssignNames();

            foreach (var variant in _postOrder)
            {
                CloneVariant(variant);
            }

            var result = new IRModule();
            foreach (var function in _module.Functions.Where(x => x is not PrimFunction))
            {
                result.Add(function);
            }

            foreach (var variant in _postOrder)
            {
                result.Add(variant.Function ?? throw new InvalidOperationException($"PrimFunction variant {variant.Name} was not materialized."));
            }

            result.Entry = entryVariant.Function;
            return result;
        }

        private FunctionVariant Discover(PrimFunction function, FunctionLayoutSignature signature)
        {
            if (!_variants.TryGetValue(function, out var perFunction))
            {
                perFunction = new();
                _variants.Add(function, perFunction);
            }

            if (perFunction.TryGetValue(signature, out var existing))
            {
                if (_active.Contains(existing))
                {
                    throw new InvalidOperationException(
                        $"Recursive PrimFunction layout specialization is not supported: {function.Name} with {signature}.");
                }

                return existing;
            }

            ValidateSignature(function, signature);
            var variant = new FunctionVariant(function, signature);
            perFunction.Add(signature, variant);
            _active.Add(variant);

            var collector = new DirectPrimFunctionCallCollector(call =>
            {
                var callee = (PrimFunction)call.Target;
                var calleeSignature = GetCallSignature(function, signature, callee, call.Arguments);
                variant.Callees.Add(call, Discover(callee, calleeSignature));
            });
            collector.Visit(function.Body);
            collector.Visit(function.Results);

            _active.Remove(variant);
            _postOrder.Add(variant);
            return variant;
        }

        private void AssignNames()
        {
            foreach (var (function, variants) in _variants)
            {
                var index = 0;
                foreach (var variant in variants.Values)
                {
                    variant.Name = index == 0 ? function.Name : $"{function.Name}_layout_{index}";
                    index++;
                }
            }
        }

        private void CloneVariant(FunctionVariant variant)
        {
            var replacements = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
            var parameters = new IVar[variant.Original.Parameters.Length];
            var bufferIndex = 0;
            for (var index = 0; index < parameters.Length; index++)
            {
                var parameter = variant.Original.Parameters[index];
                if (parameter is BufferVar bufferVar)
                {
                    var cloned = bufferVar.With(layoutAnnotation: variant.Signature.Layouts[bufferIndex++]);
                    parameters[index] = cloned;
                    replacements.Add(bufferVar, cloned);
                }
                else
                {
                    parameters[index] = parameter;
                    replacements.Add((BaseExpr)parameter, (BaseExpr)parameter);
                }
            }

            var cloner = new VariantCloner(variant, replacements);
            var body = cloner.Clone(variant.Original.Body, default);
            var results = cloner.Clone(variant.Original.Results, default);
            var function = new PrimFunction(variant.Name, variant.Original.ModuleKind, body, results, parameters)
            {
                Metadata = variant.Original.Metadata.Clone(),
                Role = variant.Original.Role,
                SchedResult = CloneSchedule(variant.Original.SchedResult, cloner.ExprMemo),
            };
            if (!CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after specializing PrimFunction {variant.Original.Name} as {variant.Name} with {variant.Signature}: " +
                    CompilerServices.Print(function));
            }

            variant.Function = function;
        }

        private static SchedFunctionResult CloneSchedule(
            SchedFunctionResult source,
            IReadOnlyDictionary<BaseExpr, BaseExpr> expressionMap)
        {
            var result = new SchedFunctionResult
            {
                DataUsage = source.DataUsage,
                ChipLocalDataPoolSize = source.ChipLocalDataPoolSize,
                BlockLocalDataPoolSize = source.BlockLocalDataPoolSize,
                DataAlign = source.DataAlign,
                OutputUsage = source.OutputUsage,
                OutputAlign = source.OutputAlign,
                IsScheduled = source.IsScheduled,
            };
            CopyRdata(source.Rdatas, result.Rdatas, expressionMap);
            CopyRdata(source.ChipLocalRdatas, result.ChipLocalRdatas, expressionMap);
            CopyRdata(source.BlockLocalRdatas, result.BlockLocalRdatas, expressionMap);
            return result;
        }

        private static void CopyRdata(
            IReadOnlyDictionary<Const, ValueRange<ulong>> source,
            IDictionary<Const, ValueRange<ulong>> destination,
            IReadOnlyDictionary<BaseExpr, BaseExpr> expressionMap)
        {
            foreach (var (constant, range) in source)
            {
                var cloned = expressionMap.TryGetValue(constant, out var value)
                    ? value as Const ?? throw new InvalidOperationException($"Cloned rdata {constant} is not a Const.")
                    : constant;
                destination.Add(cloned, range);
            }
        }

        private static FunctionLayoutSignature GetDeclaredSignature(PrimFunction function)
            => new(function.Parameters.ToArray().OfType<BufferVar>().Select(x => x.LayoutAnnotation).ToArray());

        private static FunctionLayoutSignature GetCallSignature(
            PrimFunction caller,
            FunctionLayoutSignature callerSignature,
            PrimFunction callee,
            ReadOnlySpan<BaseExpr> arguments)
        {
            if (callee.Parameters.Length != arguments.Length)
            {
                throw new InvalidOperationException(
                    $"PrimFunction call ABI mismatch for {callee.Name}: parameters={callee.Parameters.Length}, arguments={arguments.Length}.");
            }

            var callerLayouts = GetParameterLayouts(caller, callerSignature);
            var layouts = new List<BufferLayoutAnnotation>();
            for (var index = 0; index < callee.Parameters.Length; index++)
            {
                if (callee.Parameters[index] is not BufferVar formal)
                {
                    continue;
                }

                layouts.Add(GetArgumentLayout(callerLayouts, formal, arguments[index]));
            }

            return new FunctionLayoutSignature(layouts.ToArray());
        }

        private static IReadOnlyDictionary<BufferVar, BufferLayoutAnnotation> GetParameterLayouts(
            PrimFunction function,
            FunctionLayoutSignature signature)
        {
            var result = new Dictionary<BufferVar, BufferLayoutAnnotation>(ReferenceEqualityComparer.Instance);
            var index = 0;
            foreach (var parameter in function.Parameters.ToArray().OfType<BufferVar>())
            {
                result.Add(parameter, signature.Layouts[index++]);
            }

            return result;
        }

        private static BufferLayoutAnnotation GetArgumentLayout(
            IReadOnlyDictionary<BufferVar, BufferLayoutAnnotation> callerLayouts,
            BufferVar formal,
            BaseExpr argument)
        {
            if (formal.LayoutAnnotation.Kind == BufferLayoutKind.Opaque)
            {
                return BufferLayoutAnnotation.Opaque;
            }

            return argument switch
            {
                BufferVar actual => actual.LayoutAnnotation,
                TIR.Buffer actual => GetBufferLayout(callerLayouts, actual),
                _ => throw new InvalidOperationException(
                    $"Tensor BufferVar {formal.Name} expects a TIR.Buffer or BufferVar argument, got {argument.GetType().Name}."),
            };
        }

        private static BufferLayoutAnnotation GetBufferLayout(
            IReadOnlyDictionary<BufferVar, BufferLayoutAnnotation> callerLayouts,
            TIR.Buffer buffer)
        {
            if (buffer.MemSpan.Buffer.Start is BufferVar backingParameter
                && callerLayouts.TryGetValue(backingParameter, out var backingLayout))
            {
                if (backingLayout.Kind != BufferLayoutKind.ExactStrided)
                {
                    return backingLayout;
                }

                if (UsesBackingTensorLogicalLayout(buffer, backingParameter))
                {
                    return backingLayout;
                }
            }

            return BufferLayoutAnnotation.ExactStrided(buffer.Strides);
        }

        private static void ValidateSignature(PrimFunction function, FunctionLayoutSignature signature)
        {
            var parameters = function.Parameters.ToArray().OfType<BufferVar>().ToArray();
            if (parameters.Length != signature.Layouts.Count)
            {
                throw new InvalidOperationException(
                    $"PrimFunction {function.Name} layout signature arity mismatch: buffers={parameters.Length}, layouts={signature.Layouts.Count}.");
            }

            for (var index = 0; index < parameters.Length; index++)
            {
                var parameter = parameters[index];
                var layout = signature.Layouts[index];
                if (parameter.LayoutAnnotation.Kind == BufferLayoutKind.Opaque && layout.Kind != BufferLayoutKind.Opaque)
                {
                    throw new InvalidOperationException(
                        $"Opaque PrimFunction parameter {function.Name}.{parameter.Name} cannot be specialized as {layout}.");
                }

                if (parameter.LayoutAnnotation.Kind != BufferLayoutKind.Opaque && layout.Kind == BufferLayoutKind.Opaque)
                {
                    throw new InvalidOperationException(
                        $"Tensor PrimFunction parameter {function.Name}.{parameter.Name} cannot be specialized as opaque.");
                }
            }
        }

        private static bool UsesBackingTensorLogicalLayout(TIR.Buffer buffer, BufferVar backingParameter)
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

        private sealed class VariantCloner : ExprCloner<Unit>
        {
            private readonly FunctionVariant _variant;
            private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replacements;

            public VariantCloner(FunctionVariant variant, IReadOnlyDictionary<BaseExpr, BaseExpr> replacements)
            {
                _variant = variant;
                _replacements = replacements;
            }

            protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
            {
                if (_replacements.TryGetValue(expr, out var replacement))
                {
                    return replacement;
                }

                return base.DispatchVisit(expr, context);
            }

            protected override BaseExpr VisitLeafCall(Call expr, Unit context)
            {
                var arguments = expr.Arguments.ToArray().Select(x => Clone(x, context)).ToArray();
                if (expr.Target is PrimFunction)
                {
                    if (!_variant.Callees.TryGetValue(expr, out var callee))
                    {
                        throw new InvalidOperationException(
                            $"Missing discovered layout specialization for call {((PrimFunction)expr.Target).Name} in {_variant.Original.Name}.");
                    }

                    return expr.With(
                        target: callee.Function ?? throw new InvalidOperationException($"Callee variant {callee.Name} was not cloned before {_variant.Name}."),
                        arguments: arguments);
                }

                return expr.With(target: Clone(expr.Target, context), arguments: arguments);
            }

            protected override BaseExpr VisitLeafBuffer(TIR.Buffer expr, Unit context)
            {
                var cloned = (TIR.Buffer)base.VisitLeafBuffer(expr, context);
                if (expr.MemSpan.Buffer.Start is not BufferVar backingParameter
                    || !_replacements.TryGetValue(backingParameter, out var replacement)
                    || replacement is not BufferVar specializedParameter
                    || specializedParameter.LayoutAnnotation.Kind != BufferLayoutKind.ExactStrided
                    || !UsesBackingTensorLogicalLayout(expr, backingParameter))
                {
                    return cloned;
                }

                var localTensorType = specializedParameter.CheckedType switch
                {
                    DistributedType distributedType => DistributedUtility.GetDividedTensorType(distributedType),
                    TensorType tensorType => tensorType,
                    _ => throw new InvalidOperationException(
                        $"Exact BufferVar {specializedParameter.Name} does not have a tensor type."),
                };
                var localShape = localTensorType.Shape as RankedShape
                    ?? throw new InvalidOperationException($"Exact BufferVar {specializedParameter.Name} must have a ranked shape.");
                var strides = specializedParameter.LayoutAnnotation.Strides.ToArray();
                var byteSpan = BufferViewUtility.GetByteSpanSize(localShape.Dimensions, strides, cloned.ElemType.SizeInBytes);
                var physicalBuffer = cloned.MemSpan.Buffer.With(size: byteSpan);
                var memSpan = cloned.MemSpan.With(buffer: physicalBuffer, size: byteSpan);
                return cloned.With(memSpan: memSpan, strides: strides);
            }
        }

        private sealed class DirectPrimFunctionCallCollector : ExprWalker
        {
            private readonly Action<Call> _visitCall;

            public DirectPrimFunctionCallCollector(Action<Call> visitCall)
                : base(visitOtherFunctions: false)
            {
                _visitCall = visitCall;
            }

            protected override Unit VisitLeafCall(Call expr)
            {
                if (expr.Target is PrimFunction)
                {
                    _visitCall(expr);
                }

                return base.VisitLeafCall(expr);
            }
        }

        private sealed class FunctionVariant
        {
            public FunctionVariant(PrimFunction original, FunctionLayoutSignature signature)
            {
                Original = original;
                Signature = signature;
            }

            public PrimFunction Original { get; }

            public FunctionLayoutSignature Signature { get; }

            public Dictionary<Call, FunctionVariant> Callees { get; } = new(ReferenceEqualityComparer.Instance);

            public string Name { get; set; } = string.Empty;

            public PrimFunction? Function { get; set; }
        }

        private sealed record FunctionLayoutSignature(IRArray<BufferLayoutAnnotation> Layouts)
        {
            public FunctionLayoutSignature(BufferLayoutAnnotation[] layouts)
                : this(new IRArray<BufferLayoutAnnotation>(layouts))
            {
            }

            public override string ToString() => $"[{string.Join(", ", Layouts)}]";
        }
    }
}
