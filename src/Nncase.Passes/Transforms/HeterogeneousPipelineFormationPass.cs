// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Heterogeneous;
using Nncase.IR.Tensors;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Forms one persistent worker per execution module while preserving reusable
/// semantic function boundaries. Mixed-module functions are projected into one
/// function per participating module, and cross-module SSA edges become typed
/// producer/consumer channels inside the innermost projected function.
/// </summary>
public sealed class HeterogeneousPipelineFormationPass : ModulePass
{
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        if (input.Entry is not Function entry || entry.Role != FunctionRole.ModuleDispatch)
        {
            return Task.FromResult(input);
        }

        var moduleKinds = CompileSession.Target.ModuleCompilers
            .Select(compiler => compiler.ModuleKind)
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        if (moduleKinds.Length < 2)
        {
            return Task.FromResult(input);
        }

        var ownership = new OwnershipAnalysis(input, moduleKinds);
        var usedModuleKinds = ownership.CollectOwners(entry)
            .Where(moduleKinds.Contains)
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        if (usedModuleKinds.Length < 2)
        {
            return Task.FromResult(input);
        }

        var resultModuleKind = ownership.GetRequiredOwner(entry.Body, entry);
        if (resultModuleKind is null || !usedModuleKinds.Contains(resultModuleKind, StringComparer.Ordinal))
        {
            resultModuleKind = usedModuleKinds.Contains(entry.ModuleKind, StringComparer.Ordinal)
                ? entry.ModuleKind
                : usedModuleKinds[0];
        }

        var builder = new PipelineBuilder(input, entry, usedModuleKinds, resultModuleKind, ownership);
        builder.Build();
        Verify(input);
        return Task.FromResult(input);
    }

    private static void Verify(IRModule module)
    {
        if (module.Entry is not Function { Body: Call { Target: PipelineLaunch } launch } entry)
        {
            throw new InvalidOperationException("Heterogeneous pipeline formation did not produce a PipelineLaunch entry.");
        }

        if (launch[PipelineLaunch.Workers] is not IR.Tuple workers || workers.Fields.Length < 2)
        {
            throw new InvalidOperationException("Heterogeneous PipelineLaunch must contain at least two worker calls.");
        }

        var workerFunctions = workers.Fields.ToArray()
            .Select(field => field as Call ?? throw new InvalidOperationException("PipelineLaunch worker must be a function call."))
            .Select(call => call.Target as Function ?? throw new InvalidOperationException("PipelineLaunch worker target must be a Function."))
            .ToArray();
        if (workerFunctions.Any(function => function.Role != FunctionRole.PipelineWorker) ||
            workerFunctions.Select(function => function.ModuleKind).Distinct(StringComparer.Ordinal).Count() != workerFunctions.Length)
        {
            throw new InvalidOperationException("PipelineLaunch must contain exactly one PipelineWorker for each participating module.");
        }

        var executableFunctions = CollectReachableFunctions(workerFunctions);
        var endpoints = executableFunctions
            .SelectMany(function => ExprCollector.Collect(function.Body)
                .OfType<Call>()
                .Select(call => (Function: function, Call: call)))
            .Where(item => item.Call.Target is Produce or Consume)
            .ToArray();
        foreach (var channelGroup in endpoints.GroupBy(item => item.Call.Target switch
                 {
                     Produce produce => produce.ChannelId,
                     Consume consume => consume.ChannelId,
                     _ => throw new InvalidOperationException(),
                 }, StringComparer.Ordinal))
        {
            var produces = channelGroup.Where(item => item.Call.Target is Produce).ToArray();
            var consumes = channelGroup.Where(item => item.Call.Target is Consume).ToArray();
            if (produces.Length != 1 || consumes.Length != 1)
            {
                throw new InvalidOperationException(
                    $"Pipeline channel contract {channelGroup.Key} requires one producer and one consumer definition; " +
                    $"got {produces.Length} producer(s) and {consumes.Length} consumer(s).");
            }

            var produce = (Produce)produces[0].Call.Target;
            var consume = (Consume)consumes[0].Call.Target;
            if (produce.Phase != consume.Phase ||
                !Equals(GetCanonicalPayloadType(produces[0].Call[Produce.Value].CheckedType), consume.PayloadType))
            {
                throw new InvalidOperationException($"Pipeline channel contract {channelGroup.Key} endpoint contracts do not match.");
            }
        }

        if (workerFunctions.Any(function => !module.Functions.Contains(function, ReferenceEqualityComparer.Instance)))
        {
            throw new InvalidOperationException($"Pipeline entry {entry.Name} references a worker outside its IRModule.");
        }

        foreach (var function in executableFunctions)
        {
            var bodyNodes = ExprCollector.Collect(function.Body)
                .ToHashSet(ReferenceEqualityComparer.Instance);
            var freeParameters = bodyNodes
                .OfType<IVar>()
                .Where(variable => !function.Parameters.ToArray().Contains(variable, ReferenceEqualityComparer.Instance))
                .ToArray();
            if (freeParameters.Length != 0)
            {
                throw new InvalidOperationException(
                    $"Pipeline function {function.Name} contains unbound parameter(s): " +
                    string.Join(", ", freeParameters.Select(parameter => parameter.Name)));
            }
        }
    }

    private static HashSet<Function> CollectReachableFunctions(IEnumerable<Function> roots)
    {
        var functions = new HashSet<Function>(ReferenceEqualityComparer.Instance);
        var pending = new Stack<Function>(roots);
        while (pending.TryPop(out var function))
        {
            if (!functions.Add(function))
            {
                continue;
            }

            foreach (var call in ExprCollector.Collect(function.Body).OfType<Call>())
            {
                if (TryGetFunctionTarget(call.Target, out var callee))
                {
                    pending.Push(callee);
                }
            }
        }

        return functions;
    }

    private static TensorType GetCanonicalPayloadType(IRType type)
        => type switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => throw new InvalidOperationException(
                $"Heterogeneous pipeline channels require tensor payloads, got {type}."),
        };

    private static bool TryGetFunctionTarget(BaseExpr target, out Function function)
    {
        switch (target)
        {
            case Function direct:
                function = direct;
                return true;
            case FunctionWrapper { Target: Function wrapped }:
                function = wrapped;
                return true;
            default:
                function = null!;
                return false;
        }
    }

    private sealed class PipelineBuilder
    {
        private readonly IRModule _module;
        private readonly Function _entry;
        private readonly string[] _moduleKinds;
        private readonly string _resultModuleKind;
        private readonly OwnershipAnalysis _ownership;
        private readonly Dictionary<Function, ProjectionScope> _scopes =
            new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<ChannelSlot, Call> _concreteChannels = new();

        private int _nextContractId;
        private int _nextChannelInstanceId;

        public PipelineBuilder(
            IRModule module,
            Function entry,
            string[] moduleKinds,
            string resultModuleKind,
            OwnershipAnalysis ownership)
        {
            _module = module;
            _entry = entry;
            _moduleKinds = moduleKinds;
            _resultModuleKind = resultModuleKind;
            _ownership = ownership;
        }

        public IReadOnlyList<string> ModuleKinds => _moduleKinds;

        public OwnershipAnalysis Ownership => _ownership;

        public void Build()
        {
            var root = new ProjectionScope(this, _entry, isRoot: true);
            _scopes.Add(_entry, root);
            var result = root.Projectors[_resultModuleKind].Project(_entry.Body);

            var workers = new List<(Function Function, BaseExpr[] Arguments)>();
            foreach (var moduleKind in _moduleKinds)
            {
                var projector = root.Projectors[moduleKind];
                if (!projector.HasWork && !string.Equals(moduleKind, _resultModuleKind, StringComparison.Ordinal))
                {
                    continue;
                }

                var body = string.Equals(moduleKind, _resultModuleKind, StringComparison.Ordinal)
                    ? projector.Token is None
                        ? result
                        : IR.F.Heterogeneous.PipelineYield(result, (Expr)projector.Token)
                    : projector.Token;
                var function = new Function(
                    $"{_entry.Name}_{moduleKind}_worker",
                    moduleKind,
                    body,
                    projector.Parameters.ToArray())
                {
                    Role = FunctionRole.PipelineWorker,
                };
                _module.Add(function);
                workers.Add((function, projector.BuildRootArguments().ToArray()));
            }

            if (workers.Count < 2)
            {
                throw new InvalidOperationException("Heterogeneous pipeline projection produced fewer than two executable workers.");
            }

            var workerCalls = workers
                .Select(worker => (Expr)new Call(worker.Function, worker.Arguments))
                .ToArray();
            var resultWorkerIndex = workers.FindIndex(
                worker => string.Equals(worker.Function.ModuleKind, _resultModuleKind, StringComparison.Ordinal));
            if (resultWorkerIndex < 0)
            {
                throw new InvalidOperationException($"Pipeline result module {_resultModuleKind} has no projected worker.");
            }

            var launch = IR.F.Heterogeneous.PipelineLaunch(new IR.Tuple(workerCalls), resultWorkerIndex);
            var replacement = _entry.With(
                _entry.Name,
                _entry.ModuleKind,
                launch,
                _entry.Parameters.ToArray(),
                FunctionRole.ModuleDispatch);
            var entryIndex = _module.Functions.IndexOf(_entry, ReferenceEqualityComparer.Instance);
            _module.Replace(entryIndex, replacement);
        }

        public ProjectionScope GetOrBuildScope(Function function)
        {
            if (_scopes.TryGetValue(function, out var existing))
            {
                existing.Build();
                return existing;
            }

            var scope = new ProjectionScope(this, function, isRoot: false);
            _scopes.Add(function, scope);
            scope.Build();
            return scope;
        }

        public BoundaryContract CreateContract(
            Function scope,
            string sourceModuleKind,
            string destinationModuleKind,
            IRType payloadType)
        {
            var ordinal = checked(++_nextContractId);
            var scopeName = SanitizeName(scope.Name);
            var id = $"{scopeName}_{ordinal}__{sourceModuleKind}__{destinationModuleKind}";
            return new BoundaryContract(
                id,
                ordinal,
                sourceModuleKind,
                destinationModuleKind,
                payloadType);
        }

        public Call GetConcreteChannel(ChannelSlot slot)
        {
            if (_concreteChannels.TryGetValue(slot, out var existing))
            {
                return existing;
            }

            var ordinal = checked(++_nextChannelInstanceId);
            var contract = slot.Contract;
            var channel = IR.F.Heterogeneous.CreatePipelineChannel(
                $"channel_{ordinal}_{contract.SourceModuleKind}_to_{contract.DestinationModuleKind}",
                contract.SourceModuleKind,
                contract.DestinationModuleKind,
                contract.PayloadType);
            _concreteChannels.Add(slot, channel);
            return channel;
        }

        public void AddProjectedFunction(Function function) => _module.Add(function);

        private static string SanitizeName(string name)
        {
            var chars = name.Select(ch => char.IsAsciiLetterOrDigit(ch) ? ch : '_').ToArray();
            return new string(chars).Trim('_');
        }
    }

    private sealed class ProjectionScope
    {
        private readonly PipelineBuilder _builder;
        private readonly bool _isRoot;
        private readonly Dictionary<BaseExpr, Dictionary<string, BoundaryContract>> _boundaries =
            new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<BoundaryContract, ChannelSlot> _localSlots = new();
        private readonly Dictionary<Call, Dictionary<ChannelSlot, ChannelSlot>> _childSlots =
            new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<Call, Dictionary<string, ProjectedCall>> _projectedCalls =
            new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<string, ProjectedFunction> _functions = new(StringComparer.Ordinal);

        private bool _building;
        private bool _built;

        public ProjectionScope(PipelineBuilder builder, Function function, bool isRoot)
        {
            _builder = builder;
            _isRoot = isRoot;
            Function = function;
            Projectors = builder.ModuleKinds.ToDictionary(
                moduleKind => moduleKind,
                moduleKind => new WorkerProjector(this, moduleKind, isRoot),
                StringComparer.Ordinal);
        }

        public PipelineBuilder Builder => _builder;

        public Function Function { get; }

        public Dictionary<string, WorkerProjector> Projectors { get; }

        public IReadOnlyDictionary<string, ProjectedFunction> ProjectedFunctions => _functions;

        public void Build()
        {
            if (_isRoot || _built)
            {
                return;
            }

            if (_building)
            {
                throw new InvalidOperationException($"Heterogeneous function call graph contains recursion involving {Function.Name}.");
            }

            _building = true;
            try
            {
                var outputFields = GetOutputFields(Function.Body);
                for (var fieldIndex = 0; fieldIndex < outputFields.Length; fieldIndex++)
                {
                    var field = outputFields[fieldIndex];
                    var owner = GetOwner(field);
                    if (owner is null)
                    {
                        if (field is IVar || field is Const or None)
                        {
                            continue;
                        }

                        throw new InvalidOperationException(
                            $"Cannot determine one execution owner for output {fieldIndex} of mixed function {Function.Name}. " +
                            "Return tuple fields explicitly or assign the producing operation before pipeline formation.");
                    }

                    if (!Projectors.TryGetValue(owner, out var projector))
                    {
                        throw new InvalidOperationException(
                            $"Function {Function.Name} output {fieldIndex} is owned by unknown module {owner}.");
                    }

                    projector.AddOutput(fieldIndex, projector.Project(field));
                }

                foreach (var (moduleKind, projector) in Projectors)
                {
                    if (!projector.HasWork)
                    {
                        continue;
                    }

                    var body = projector.BuildFunctionBody();
                    var projected = new Function(
                        $"{Function.Name}_{moduleKind}",
                        moduleKind,
                        body,
                        projector.Parameters.ToArray())
                    {
                        Role = FunctionRole.PipelineProjection,
                    };
                    projected.Metadata = Function.Metadata.Clone();
                    _builder.AddProjectedFunction(projected);
                    _functions.Add(
                        moduleKind,
                        new ProjectedFunction(
                            projected,
                            projector.Bindings.ToArray(),
                            projector.OutputFieldIndices.ToArray(),
                            projector.HasPipelineEffects));
                }

                if (_functions.Count == 0)
                {
                    throw new InvalidOperationException($"Mixed function {Function.Name} has no executable projection.");
                }

                _built = true;
            }
            finally
            {
                _building = false;
            }
        }

        public string? GetOwner(BaseExpr expr) => _builder.Ownership.GetOwner(expr, Function);

        public BaseExpr ResolveForeignValue(
            BaseExpr value,
            string sourceModuleKind,
            WorkerProjector destination)
        {
            if (value.CheckedType is TupleType tupleType)
            {
                var fields = Enumerable.Range(0, tupleType.Fields.Count)
                    .Select(fieldIndex => ResolveForeignTupleField(
                        value,
                        fieldIndex,
                        sourceModuleKind,
                        destination))
                    .ToArray();
                return new IR.Tuple(fields);
            }

            if (!_boundaries.TryGetValue(value, out var byDestination))
            {
                byDestination = new Dictionary<string, BoundaryContract>(StringComparer.Ordinal);
                _boundaries.Add(value, byDestination);
            }

            if (!byDestination.TryGetValue(destination.ModuleKind, out var contract))
            {
                var payloadType = GetCanonicalPayloadType(value.CheckedType);
                contract = _builder.CreateContract(
                    Function,
                    sourceModuleKind,
                    destination.ModuleKind,
                    payloadType);
                byDestination.Add(destination.ModuleKind, contract);

                var source = Projectors[sourceModuleKind];
                var sourceValue = source.Project(value);
                source.AppendProduce(contract, GetLocalSlot(contract), sourceValue);
            }

            return destination.AppendConsume(contract, GetLocalSlot(contract));
        }

        public BaseExpr ResolveForeignTupleField(
            BaseExpr tupleValue,
            int fieldIndex,
            string sourceModuleKind,
            WorkerProjector destination)
        {
            if (tupleValue.CheckedType is not TupleType tupleType ||
                fieldIndex < 0 ||
                fieldIndex >= tupleType.Fields.Count)
            {
                throw new InvalidOperationException(
                    $"Cannot project field {fieldIndex} from heterogeneous boundary value {tupleValue.CheckedType}.");
            }

            var field = tupleValue is IR.Tuple tuple
                ? tuple.Fields[fieldIndex]
                : IR.F.Tensors.GetItem(tupleValue, fieldIndex);
            if (!CompilerServices.InferenceType(field))
            {
                throw new InvalidOperationException(
                    $"Type inference failed for heterogeneous tuple field {fieldIndex} of {tupleValue.CheckedType}.");
            }

            return ResolveForeignValue(field, sourceModuleKind, destination);
        }

        public BaseExpr ProjectStructuredCall(Call call, WorkerProjector requester)
        {
            var fields = GetOutputFields(GetStructuredCallee(call).Body);
            if (fields.Length != 1)
            {
                throw new InvalidOperationException(
                    $"Tuple-valued dispatch call {GetStructuredCallee(call).Name} must be consumed through GetItem.");
            }

            return ProjectStructuredCallField(call, 0, requester);
        }

        public BaseExpr ProjectStructuredCallField(Call call, int fieldIndex, WorkerProjector requester)
        {
            var callee = GetStructuredCallee(call);
            var fields = GetOutputFields(callee.Body);
            if (fieldIndex < 0 || fieldIndex >= fields.Length)
            {
                throw new InvalidOperationException($"Function {callee.Name} has no output field {fieldIndex}.");
            }

            var outputExpr = fields[fieldIndex];
            if (TryGetPassthroughArgument(callee, call, outputExpr, out var passthrough))
            {
                return requester.Project(passthrough);
            }

            var owner = _builder.Ownership.GetOwner(outputExpr, callee)
                ?? throw new InvalidOperationException(
                    $"Cannot determine owner of output {fieldIndex} from mixed function {callee.Name}.");
            EnsureProjectedCalls(call, callee);
            if (string.Equals(owner, requester.ModuleKind, StringComparison.Ordinal))
            {
                return GetProjectedOutput(call, owner, fieldIndex);
            }

            return ResolveForeignValue(
                fieldIndex == 0 && fields.Length == 1
                    ? call
                    : IR.F.Tensors.GetItem(call, fieldIndex),
                owner,
                requester);
        }

        public ChannelSlot MapChildSlot(Call call, ChannelSlot childSlot)
        {
            if (!_childSlots.TryGetValue(call, out var slots))
            {
                slots = new Dictionary<ChannelSlot, ChannelSlot>();
                _childSlots.Add(call, slots);
            }

            if (!slots.TryGetValue(childSlot, out var mapped))
            {
                mapped = new ChannelSlot(childSlot.Contract, $"{Function.Name}/{call.GetHashCode():x}/{childSlot.Path}");
                slots.Add(childSlot, mapped);
            }

            return mapped;
        }

        private void EnsureProjectedCalls(Call call, Function callee)
        {
            if (_projectedCalls.ContainsKey(call))
            {
                return;
            }

            var child = _builder.GetOrBuildScope(callee);
            var calls = new Dictionary<string, ProjectedCall>(StringComparer.Ordinal);
            _projectedCalls.Add(call, calls);
            foreach (var (moduleKind, projectedFunction) in child.ProjectedFunctions)
            {
                var parent = Projectors[moduleKind];
                var arguments = new List<BaseExpr>(projectedFunction.Bindings.Length);
                foreach (var binding in projectedFunction.Bindings)
                {
                    arguments.Add(binding switch
                    {
                        OriginalParameterBinding original => parent.Project(call.Arguments[original.ParameterIndex]),
                        ChannelParameterBinding channel => parent.GetChannel(MapChildSlot(call, channel.Slot)),
                        DependencyParameterBinding => parent.GetDependency(),
                        _ => throw new InvalidOperationException($"Unknown projection binding {binding.GetType().Name}."),
                    });
                }

                var projectedCall = new Call(projectedFunction.Function, arguments.ToArray());
                if (!CompilerServices.InferenceType(projectedCall))
                {
                    throw new InvalidOperationException($"Type inference failed for projected call {projectedFunction.Function.Name}.");
                }

                calls.Add(moduleKind, new ProjectedCall(projectedCall, projectedFunction));
                parent.AppendProjectedCall(projectedCall, projectedFunction.HasPipelineEffects);
            }
        }

        private BaseExpr GetProjectedOutput(Call originalCall, string moduleKind, int fieldIndex)
        {
            if (!_projectedCalls.TryGetValue(originalCall, out var calls) ||
                !calls.TryGetValue(moduleKind, out var projected))
            {
                throw new InvalidOperationException(
                    $"Dispatch call {GetStructuredCallee(originalCall).Name} has no {moduleKind} projection.");
            }

            var compactIndex = Array.IndexOf(projected.Function.OutputFieldIndices, fieldIndex);
            if (compactIndex < 0)
            {
                throw new InvalidOperationException(
                    $"Projection {projected.Function.Function.Name} does not return original output field {fieldIndex}.");
            }

            if (projected.Function.OutputFieldIndices.Length == 1)
            {
                return projected.Call;
            }

            var item = IR.F.Tensors.GetItem(projected.Call, compactIndex);
            if (!CompilerServices.InferenceType(item))
            {
                throw new InvalidOperationException(
                    $"Type inference failed for projected output {compactIndex} of {projected.Function.Function.Name}.");
            }

            return item;
        }

        private ChannelSlot GetLocalSlot(BoundaryContract contract)
        {
            if (!_localSlots.TryGetValue(contract, out var slot))
            {
                slot = new ChannelSlot(contract, $"{Function.Name}/{contract.Id}");
                _localSlots.Add(contract, slot);
            }

            return slot;
        }

        private static Function GetStructuredCallee(Call call)
        {
            if (!TryGetFunctionTarget(call.Target, out var callee) || callee.Role != FunctionRole.ModuleDispatch)
            {
                throw new InvalidOperationException("Expected a ModuleDispatch function call.");
            }

            return callee;
        }

        private static BaseExpr[] GetOutputFields(BaseExpr body)
            => body is IR.Tuple tuple ? tuple.Fields.ToArray() : new[] { body };

        private static bool TryGetPassthroughArgument(
            Function callee,
            Call call,
            BaseExpr output,
            out BaseExpr argument)
        {
            for (var index = 0; index < callee.Parameters.Length; index++)
            {
                if (ReferenceEquals(output, callee.Parameters[index]))
                {
                    argument = call.Arguments[index];
                    return true;
                }
            }

            if (output is Const or None)
            {
                argument = output;
                return true;
            }

            argument = null!;
            return false;
        }
    }

    private sealed class WorkerProjector : ExprCloner<Unit>
    {
        private readonly ProjectionScope _scope;
        private readonly bool _isRoot;
        private readonly Dictionary<BaseExpr, IVar> _parameterMap = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<ChannelSlot, Var> _channelParameters = new();
        private readonly Dictionary<BoundaryContract, BaseExpr> _consumedValues = new();
        private readonly List<BaseExpr> _rootArguments = new();
        private readonly List<BaseExpr> _outputs = new();
        private readonly List<int> _outputFieldIndices = new();

        private Var? _dependencyParameter;

        public WorkerProjector(ProjectionScope scope, string moduleKind, bool isRoot)
        {
            _scope = scope;
            _isRoot = isRoot;
            ModuleKind = moduleKind;
            Token = None.Default;
            Parameters = new List<IVar>();
            Bindings = new List<ProjectionParameterBinding>();
        }

        public string ModuleKind { get; }

        public BaseExpr Token { get; private set; }

        public List<IVar> Parameters { get; }

        public List<ProjectionParameterBinding> Bindings { get; }

        public IReadOnlyList<int> OutputFieldIndices => _outputFieldIndices;

        public bool HasPipelineEffects { get; private set; }

        public bool HasWork => HasPipelineEffects || _outputs.Count != 0;

        public BaseExpr Project(BaseExpr expr) => Visit(expr, Unit.Default);

        public void AddOutput(int fieldIndex, BaseExpr value)
        {
            _outputFieldIndices.Add(fieldIndex);
            _outputs.Add(value);
        }

        public void AppendProduce(BoundaryContract contract, ChannelSlot slot, BaseExpr value)
        {
            var channel = GetChannel(slot);
            Token = IR.F.Heterogeneous.Produce(
                (Expr)channel,
                (Expr)value,
                (Expr)GetDependency(),
                contract.Id,
                contract.Phase);
            HasPipelineEffects = true;
        }

        public BaseExpr AppendConsume(BoundaryContract contract, ChannelSlot slot)
        {
            if (_consumedValues.TryGetValue(contract, out var existing))
            {
                return existing;
            }

            var channel = GetChannel(slot);
            var consume = IR.F.Heterogeneous.Consume(
                (Expr)channel,
                (Expr)GetDependency(),
                contract.Id,
                contract.Phase,
                contract.PayloadType);
            Token = IR.F.Heterogeneous.PipelineToken(consume);
            HasPipelineEffects = true;
            _consumedValues.Add(contract, consume);
            return consume;
        }

        public BaseExpr GetDependency()
        {
            if (Token is not None)
            {
                return Token;
            }

            if (_isRoot)
            {
                return None.Default;
            }

            _dependencyParameter ??= RegisterDependencyParameter();
            Token = _dependencyParameter;
            return Token;
        }

        public BaseExpr GetChannel(ChannelSlot slot)
        {
            if (_channelParameters.TryGetValue(slot, out var existing))
            {
                return existing;
            }

            var parameter = new Var(
                $"pipeline_channel_{Parameters.Count}",
                TensorType.Scalar(new ReferenceType(new PipelineChannelType())));
            _channelParameters.Add(slot, parameter);
            Parameters.Add(parameter);
            Bindings.Add(new ChannelParameterBinding(parameter, slot));
            if (_isRoot)
            {
                _rootArguments.Add(_scope.Builder.GetConcreteChannel(slot));
            }

            return parameter;
        }

        public void AppendProjectedCall(Call call, bool hasPipelineEffects)
        {
            if (!hasPipelineEffects)
            {
                return;
            }

            Token = call.CheckedType is NoneType
                ? call
                : IR.F.Heterogeneous.PipelineToken(call);
            HasPipelineEffects = true;
        }

        public BaseExpr BuildFunctionBody()
        {
            BaseExpr value = _outputs.Count switch
            {
                0 => Token,
                1 => _outputs[0],
                _ => new IR.Tuple(_outputs.ToArray()),
            };
            return _outputs.Count != 0 && Token is not None
                ? IR.F.Heterogeneous.PipelineYield(value, (Expr)Token)
                : value;
        }

        public IReadOnlyList<BaseExpr> BuildRootArguments()
        {
            if (!_isRoot || _rootArguments.Count != Parameters.Count)
            {
                throw new InvalidOperationException(
                    $"Root projection {ModuleKind} argument ABI does not match its parameters.");
            }

            return _rootArguments;
        }

        protected internal override BaseExpr VisitCall(Call expr, Unit context)
        {
            if (expr.Target is GetItem &&
                expr[GetItem.Input] is Call inputCall &&
                TryGetFunctionTarget(inputCall.Target, out var tupleCallee) &&
                tupleCallee.Role == FunctionRole.ModuleDispatch &&
                expr[GetItem.Index] is DimConst index)
            {
                return _scope.ProjectStructuredCallField(
                    inputCall,
                    checked((int)index.Value),
                    this);
            }

            if (TryGetFunctionTarget(expr.Target, out var callee) &&
                callee.Role == FunctionRole.ModuleDispatch)
            {
                return _scope.ProjectStructuredCall(expr, this);
            }

            return base.VisitCall(expr, context);
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (expr is IVar parameter && IsFunctionParameter(parameter))
            {
                var owner = _isRoot ? null : _scope.GetOwner((BaseExpr)parameter);
                if (owner is not null && !string.Equals(owner, ModuleKind, StringComparison.Ordinal))
                {
                    return _scope.ResolveForeignValue((BaseExpr)parameter, owner, this);
                }

                return GetOriginalParameter(parameter);
            }

            var expressionOwner = _scope.GetOwner(expr);
            if (expressionOwner is not null && !string.Equals(expressionOwner, ModuleKind, StringComparison.Ordinal))
            {
                return _scope.ResolveForeignValue(expr, expressionOwner, this);
            }

            return base.DispatchVisit(expr, context);
        }

        private bool IsFunctionParameter(IVar parameter)
            => _scope.Function.Parameters.ToArray().Any(candidate => ReferenceEquals(candidate, parameter));

        private BaseExpr GetOriginalParameter(IVar parameter)
        {
            if (_parameterMap.TryGetValue((BaseExpr)parameter, out var existing))
            {
                return (BaseExpr)existing;
            }

            var projected = parameter is Var var ? var.With() : parameter;
            _parameterMap.Add((BaseExpr)parameter, projected);
            Parameters.Add(projected);
            var parameterArray = _scope.Function.Parameters.ToArray();
            var parameterIndex = Array.FindIndex(parameterArray, candidate => ReferenceEquals(candidate, parameter));
            if (parameterIndex < 0)
            {
                throw new InvalidOperationException($"Parameter {parameter.Name} does not belong to {_scope.Function.Name}.");
            }

            Bindings.Add(new OriginalParameterBinding(projected, parameterIndex));
            if (_isRoot)
            {
                _rootArguments.Add((BaseExpr)parameter);
            }

            return (BaseExpr)projected;
        }

        private Var RegisterDependencyParameter()
        {
            var parameter = new Var("pipeline_dependency", NoneType.Default);
            Parameters.Add(parameter);
            Bindings.Add(new DependencyParameterBinding(parameter));
            return parameter;
        }
    }

    private sealed class OwnershipAnalysis
    {
        private readonly HashSet<string> _moduleKinds;
        private readonly HashSet<Function> _functions;
        private readonly Dictionary<Function, Dictionary<IVar, string?>> _parameterOwners =
            new(ReferenceEqualityComparer.Instance);

        public OwnershipAnalysis(IRModule module, IEnumerable<string> moduleKinds)
        {
            _moduleKinds = moduleKinds.ToHashSet(StringComparer.Ordinal);
            _functions = new HashSet<Function>(
                module.Functions.OfType<Function>(),
                ReferenceEqualityComparer.Instance);
            foreach (var function in _functions)
            {
                var parameterOwners = new Dictionary<IVar, string?>(ReferenceEqualityComparer.Instance);
                foreach (var parameter in function.Parameters)
                {
                    parameterOwners.Add(parameter, null);
                }

                _parameterOwners.Add(
                    function,
                    parameterOwners);
            }

            InferParameterOwners();
        }

        public IEnumerable<string> CollectOwners(Function function)
        {
            var visited = new HashSet<Function>(ReferenceEqualityComparer.Instance);
            return Collect(function, visited).ToArray();

            IEnumerable<string> Collect(Function current, HashSet<Function> active)
            {
                if (!active.Add(current))
                {
                    yield break;
                }

                var collector = new LocalExpressionCollector();
                collector.Visit(current.Body);
                foreach (var expr in collector.Expressions)
                {
                    var owner = GetOwner(expr, current);
                    if (owner is not null)
                    {
                        yield return owner;
                    }
                }

                foreach (var call in collector.Expressions.OfType<Call>())
                {
                    if (TryGetFunctionTarget(call.Target, out var callee) && _functions.Contains(callee))
                    {
                        foreach (var owner in Collect(callee, active))
                        {
                            yield return owner;
                        }
                    }
                }
            }
        }

        public string? GetRequiredOwner(BaseExpr expr, Function owner)
        {
            var result = GetOwner(expr, owner);
            if (result is not null)
            {
                return result;
            }

            if (expr is IR.Tuple tuple)
            {
                var owners = tuple.Fields.ToArray()
                    .Select(field => GetOwner(field, owner))
                    .Where(candidate => candidate is not null)
                    .Distinct(StringComparer.Ordinal)
                    .ToArray();
                return owners.Length == 1 ? owners[0] : null;
            }

            return null;
        }

        public string? GetOwner(BaseExpr expr, Function scope)
        {
            if (expr is Call { Target: GetItem } getItem &&
                getItem[GetItem.Input] is Call tupleCall &&
                TryGetFunctionTarget(tupleCall.Target, out var tupleCallee) &&
                getItem[GetItem.Index] is DimConst index)
            {
                return tupleCallee.Role == FunctionRole.ModuleDispatch
                    ? GetFunctionOutputOwner(tupleCallee, checked((int)index.Value))
                    : tupleCallee.ModuleKind;
            }

            return expr switch
            {
                IVar parameter when _parameterOwners.TryGetValue(scope, out var owners) && owners.TryGetValue(parameter, out var parameterOwner)
                    => parameterOwner,
                Call { Target: Function callee } when callee.Role != FunctionRole.ModuleDispatch
                    => callee.ModuleKind,
                Call { Target: FunctionWrapper { Target: Function callee } } when callee.Role != FunctionRole.ModuleDispatch
                    => callee.ModuleKind,
                Call { Target: Function callee }
                    => GetFunctionOutputOwner(callee, 0),
                Call { Target: FunctionWrapper { Target: Function callee } }
                    => GetFunctionOutputOwner(callee, 0),
                Call { Target: Op } call when !string.IsNullOrWhiteSpace(call.Metadata.ExecutionModuleKind)
                    => call.Metadata.ExecutionModuleKind,
                BaseFunction => null,
                IVar => null,
                Const => null,
                None => null,
                IR.Tuple tuple => MergeOwners(tuple.Fields.ToArray().Select(field => GetOwner(field, scope))),
                _ => MergeOwners(expr.Operands.ToArray().Select(operand => GetOwner(operand, scope))),
            };
        }

        private void InferParameterOwners()
        {
            var ownerSets = new Dictionary<Function, Dictionary<IVar, HashSet<string>>>(
                ReferenceEqualityComparer.Instance);
            foreach (var function in _functions)
            {
                var parameterOwnerSets = new Dictionary<IVar, HashSet<string>>(
                    ReferenceEqualityComparer.Instance);
                foreach (var parameter in function.Parameters)
                {
                    parameterOwnerSets.Add(parameter, new HashSet<string>(StringComparer.Ordinal));
                }

                ownerSets.Add(function, parameterOwnerSets);
            }

            foreach (var caller in _functions)
            {
                var collector = new LocalExpressionCollector();
                collector.Visit(caller.Body);
                foreach (var call in collector.Expressions.OfType<Call>())
                {
                    if (!TryGetFunctionTarget(call.Target, out var callee) || !_functions.Contains(callee))
                    {
                        continue;
                    }

                    for (var index = 0; index < Math.Min(call.Arguments.Length, callee.Parameters.Length); index++)
                    {
                        var owner = GetOwner(call.Arguments[index], caller);
                        if (owner is not null)
                        {
                            ownerSets[callee][callee.Parameters[index]].Add(owner);
                        }
                    }
                }
            }

            foreach (var function in _functions)
            {
                foreach (var parameter in function.Parameters)
                {
                    var candidates = ownerSets[function][parameter];
                    if (candidates.Count > 1 && function.Role == FunctionRole.ModuleDispatch)
                    {
                        throw new InvalidOperationException(
                            $"Mixed function {function.Name} parameter {parameter.Name} is supplied by multiple modules " +
                            $"({string.Join(", ", candidates)}). Split its semantic call contract before heterogeneous projection.");
                    }

                    _parameterOwners[function][parameter] = candidates.Count == 1 ? candidates.Single() : null;
                }
            }
        }

        private string? GetFunctionOutputOwner(Function function, int fieldIndex)
        {
            var fields = function.Body is IR.Tuple tuple ? tuple.Fields.ToArray() : new[] { function.Body };
            return fieldIndex >= 0 && fieldIndex < fields.Length
                ? GetOwner(fields[fieldIndex], function)
                : null;
        }

        private string? MergeOwners(IEnumerable<string?> owners)
        {
            var distinct = owners
                .Where(owner => owner is not null && _moduleKinds.Contains(owner))
                .Distinct(StringComparer.Ordinal)
                .ToArray();
            return distinct.Length == 1 ? distinct[0] : null;
        }

        private sealed class LocalExpressionCollector : ExprWalker
        {
            public List<BaseExpr> Expressions { get; } = new();

            protected override Unit DefaultVisitLeaf(BaseExpr expr)
            {
                Expressions.Add(expr);
                return base.DefaultVisitLeaf(expr);
            }
        }
    }

    private sealed record BoundaryContract(
        string Id,
        int Phase,
        string SourceModuleKind,
        string DestinationModuleKind,
        IRType PayloadType);

    private sealed record ChannelSlot(BoundaryContract Contract, string Path);

    private abstract record ProjectionParameterBinding(IVar Parameter);

    private sealed record OriginalParameterBinding(IVar Parameter, int ParameterIndex)
        : ProjectionParameterBinding(Parameter);

    private sealed record ChannelParameterBinding(IVar Parameter, ChannelSlot Slot)
        : ProjectionParameterBinding(Parameter);

    private sealed record DependencyParameterBinding(IVar Parameter)
        : ProjectionParameterBinding(Parameter);

    private sealed record ProjectedFunction(
        Function Function,
        ProjectionParameterBinding[] Bindings,
        int[] OutputFieldIndices,
        bool HasPipelineEffects);

    private sealed record ProjectedCall(Call Call, ProjectedFunction Function);
}
