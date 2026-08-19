// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Forwards terminal canonical tensor stores into caller-allocated outputs.
/// </summary>
public sealed class ForwardTerminalStoreDestinationsPass : ModulePass
{
    private readonly string _moduleKind;

    public ForwardTerminalStoreDestinationsPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        foreach (var function in input.Functions
                     .OfType<PrimFunction>()
                     .Where(function => function.ModuleKind == _moduleKind))
        {
            var rewriter = new DestinationForwardingRewriter(function);
            rewriter.Rewrite(function);
            if (rewriter.IsMutated && !CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after forwarding terminal stores in {function.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private sealed class DestinationForwardingRewriter : ExprRewriter
    {
        private readonly IReadOnlySet<BaseExpr> _functionNodes;
        private readonly IReadOnlyDictionary<TIR.Buffer, BufferAccess[]> _bufferAccesses;
        private readonly IReadOnlyDictionary<PhysicalBuffer, TIR.Buffer[]> _physicalAliases;

        public DestinationForwardingRewriter(PrimFunction function)
            : base(visitOtherFunctions: false)
        {
            var expressions = ExprCollector.Collect(function).ToArray();
            _functionNodes = expressions.ToHashSet(new ReferenceEqualityComparer<BaseExpr>());
            _bufferAccesses = CollectBufferAccesses(expressions.OfType<Call>());
            _physicalAliases = expressions
                .OfType<TIR.Buffer>()
                .Distinct(new ReferenceEqualityComparer<TIR.Buffer>())
                .GroupBy(buffer => buffer.MemSpan.Buffer, new ReferenceEqualityComparer<PhysicalBuffer>())
                .ToDictionary(
                    group => group.Key,
                    group => group.ToArray(),
                    new ReferenceEqualityComparer<PhysicalBuffer>());
        }

        protected override BaseExpr RewriteLeafSequential(Sequential expr)
        {
            var fields = expr.Fields.ToArray();
            var directCallIndices = fields
                .Select((field, index) => (field, index))
                .Where(item => item.field is Call)
                .ToDictionary(
                    item => (Call)item.field,
                    item => item.index,
                    new ReferenceEqualityComparer<Call>());
            var forwardedStores = new HashSet<Call>(new ReferenceEqualityComparer<Call>());
            var writerReplacements = new Dictionary<Call, Dictionary<int, TIR.Buffer>>(
                new ReferenceEqualityComparer<Call>());

            foreach (var (field, storeIndex) in fields.Select((field, index) => (field, index)))
            {
                if (field is not Call { Target: TIR.NTT.TensorStore tensorStore } storeCall ||
                    storeCall[TIR.NTT.TensorStore.Src] is not TIR.Buffer source ||
                    storeCall[TIR.NTT.TensorStore.Dest] is not TIR.Buffer destination ||
                    !TryCreateForwardedBuffer(
                        tensorStore,
                        storeCall,
                        source,
                        destination,
                        directCallIndices,
                        storeIndex,
                        out var writer,
                        out var writerArgumentIndex,
                        out var forwarded))
                {
                    continue;
                }

                if (!writerReplacements.TryGetValue(writer, out var replacements))
                {
                    replacements = new Dictionary<int, TIR.Buffer>();
                    writerReplacements.Add(writer, replacements);
                }

                if (!replacements.TryAdd(writerArgumentIndex, forwarded))
                {
                    throw new InvalidOperationException(
                        $"Terminal store forwarding selected writer operand {writerArgumentIndex} more than once.");
                }

                forwardedStores.Add(storeCall);
            }

            if (forwardedStores.Count == 0)
            {
                return expr;
            }

            var rewrittenFields = new List<Expr>(fields.Length - forwardedStores.Count);
            foreach (var field in fields)
            {
                if (field is Call call && forwardedStores.Contains(call))
                {
                    continue;
                }

                if (field is Call writer && writerReplacements.TryGetValue(writer, out var replacements))
                {
                    var arguments = writer.Arguments.ToArray();
                    foreach (var (argumentIndex, replacement) in replacements)
                    {
                        arguments[argumentIndex] = replacement;
                    }

                    rewrittenFields.Add(writer.With(arguments: arguments));
                }
                else
                {
                    rewrittenFields.Add(field);
                }
            }

            SetMutated();
            return expr.With(fields: rewrittenFields.ToArray());
        }

        private bool TryCreateForwardedBuffer(
            TIR.NTT.TensorStore tensorStore,
            Call storeCall,
            TIR.Buffer source,
            TIR.Buffer destination,
            IReadOnlyDictionary<Call, int> directCallIndices,
            int storeIndex,
            out Call writer,
            out int writerArgumentIndex,
            out TIR.Buffer forwarded)
        {
            writer = null!;
            writerArgumentIndex = -1;
            forwarded = null!;

            if (!IsCompatibleTerminalStore(tensorStore, source, destination) ||
                !_physicalAliases.TryGetValue(source.MemSpan.Buffer, out var sourceAliases) ||
                sourceAliases.Length != 1 ||
                !_physicalAliases.TryGetValue(destination.MemSpan.Buffer, out var destinationAliases) ||
                destinationAliases.Length != 1 ||
                !_bufferAccesses.TryGetValue(source, out var accesses) ||
                accesses.Length != 2 ||
                !_bufferAccesses.TryGetValue(destination, out var destinationAccesses) ||
                destinationAccesses.Length != 1 ||
                !ReferenceEquals(destinationAccesses[0].Call, storeCall))
            {
                return false;
            }

            var storeSourceAccess = accesses.SingleOrDefault(access =>
                ReferenceEquals(access.Call, storeCall) &&
                MemoryEffectUtility.GetPhysicalBufferAccessMode(access.Effect) == MemoryAccessMode.Read);
            var writerAccesses = accesses.Where(access =>
                !ReferenceEquals(access.Call, storeCall) &&
                MemoryEffectUtility.GetPhysicalBufferAccessMode(access.Effect) == MemoryAccessMode.Write &&
                access.Effect.Kind == MemoryEffectKind.Direct).ToArray();
            if (storeSourceAccess is null ||
                writerAccesses.Length != 1 ||
                writerAccesses[0].Call.Target is not TIR.NTT.NTTKernelOp ||
                !directCallIndices.TryGetValue(writerAccesses[0].Call, out var writerIndex) ||
                writerIndex >= storeIndex)
            {
                return false;
            }

            var activeUsers = source.Users.Where(_functionNodes.Contains).ToArray();
            if (activeUsers.Length != 2 ||
                !activeUsers.Any(user => ReferenceEquals(user, storeCall)) ||
                !activeUsers.Any(user => ReferenceEquals(user, writerAccesses[0].Call)))
            {
                return false;
            }

            var storeDestinationEffect = GetSingleBufferEffect(storeCall, destination);
            if (storeDestinationEffect is null ||
                MemoryEffectUtility.GetPhysicalBufferAccessMode(storeDestinationEffect.Value) != MemoryAccessMode.Write ||
                storeDestinationEffect.Value.Kind != MemoryEffectKind.Direct ||
                writerAccesses[0].Effect.Scope != storeDestinationEffect.Value.Scope ||
                writerAccesses[0].Effect.AccessDomain != storeDestinationEffect.Value.AccessDomain)
            {
                return false;
            }

            var forwardedSpan = new MemSpan(
                destination.MemSpan.Buffer,
                destination.MemSpan.Start,
                source.MemSpan.Size);
            writer = writerAccesses[0].Call;
            writerArgumentIndex = writerAccesses[0].ArgumentIndex;
            forwarded = source.With(
                name: $"{source.Name}_output",
                memSpan: forwardedSpan);
            return true;
        }

        private bool IsCompatibleTerminalStore(
            TIR.NTT.TensorStore tensorStore,
            TIR.Buffer source,
            TIR.Buffer destination)
        {
            if (destination.MemSpan.Buffer.Location != MemoryLocation.Output ||
                destination.MemSpan.Buffer.Start is not BufferVar destinationVar ||
                destinationVar.Role != BufferVarRole.Output ||
                destinationVar.Location != MemoryLocation.Output ||
                destination.DistributedType is not null ||
                source.DistributedType is not { Partial: null } sourceType ||
                source.DistributedStorageKind != DistributedBufferStorageKind.CanonicalGlobal ||
                source.StorageEncoding is not null ||
                source.StagedLayout is not null ||
                destination.StorageEncoding is not null ||
                destination.StagedLayout is not null ||
                !sourceType.TensorType.Equals(
                    new TensorType(destination.ElemType, destination.Dimensions.ToArray())) ||
                !source.Strides.SequenceEqual(destination.Strides) ||
                !tensorStore.NdSbp.SequenceEqual(sourceType.AxisPolicies) ||
                !tensorStore.Placement.Equals(sourceType.Placement))
            {
                return false;
            }

            var requiredBytes = GetMaximum(source.MemSpan.Size);
            var availableBytes = GetMaximum(destination.MemSpan.Size);
            var requiredAlignment = source.MemSpan.Buffer.Alignment;
            return requiredBytes <= availableBytes &&
                   destination.MemSpan.Start.IsFixed &&
                   destination.MemSpan.Start.FixedValue % requiredAlignment == 0 &&
                   destination.MemSpan.Buffer.Alignment >= requiredAlignment;
        }

        private long GetMaximum(Dimension dimension)
            => CompilerServices.GetMaxShape([dimension])[0];

        private MemoryEffect? GetSingleBufferEffect(Call call, TIR.Buffer buffer)
        {
            var effects = new List<MemoryEffect>();
            MemoryEffectUtility.VisitCallEffects(call, (argument, _, _, effect) =>
            {
                if (ReferenceEquals(argument, buffer))
                {
                    effects.Add(effect);
                }
            });
            return effects.Count == 1 ? effects[0] : null;
        }

        private IReadOnlyDictionary<TIR.Buffer, BufferAccess[]> CollectBufferAccesses(
            IEnumerable<Call> calls)
        {
            var accesses = new Dictionary<TIR.Buffer, List<BufferAccess>>(
                new ReferenceEqualityComparer<TIR.Buffer>());
            foreach (var call in calls.Where(call => call.Target is Op))
            {
                MemoryEffectUtility.VisitCallEffects(call, (argument, _, argumentIndex, effect) =>
                {
                    if (argument is not TIR.Buffer buffer)
                    {
                        return;
                    }

                    if (!accesses.TryGetValue(buffer, out var bufferAccesses))
                    {
                        bufferAccesses = new List<BufferAccess>();
                        accesses.Add(buffer, bufferAccesses);
                    }

                    bufferAccesses.Add(new BufferAccess(call, argumentIndex, effect));
                });
            }

            return accesses.ToDictionary(
                item => item.Key,
                item => item.Value.ToArray(),
                new ReferenceEqualityComparer<TIR.Buffer>());
        }

        private sealed record BufferAccess(Call Call, int ArgumentIndex, MemoryEffect Effect);
    }
}
