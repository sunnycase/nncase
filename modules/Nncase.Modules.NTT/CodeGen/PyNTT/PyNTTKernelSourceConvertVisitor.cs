// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reactive;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.CodeGen.PyNTT;

internal sealed class PyNTTKernelSourceConvertVisitor : ExprFunctor<Unit, Unit>
{
    private const int MaxGeneratedIdentifierLength = 220;
    private const string PoolStrideElementsSuffix = "_pool_stride_elements";

    private static readonly long[] ElementwiseBlockSizeSearchSpace = { 128, 256, 512, 1024 };
    private static readonly Regex InputReferenceRegex = new(@"\binput(?<index>\d+)(?<suffix>_pool_stride_elements|_(?:scalar_)?stride\d+)?\b", RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly Regex AbiArgumentReferenceRegex = new(@"\b(?:input|output)\d+(?:_pool_stride_elements|_(?:scalar_)?stride\d+)?\b", RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly Regex AbiArgumentSuffixRegex = new(@"_(?:pool_stride_elements|(?:scalar_)?stride\d+)$", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private readonly List<GeneratedKernelMetadata> _generatedKernels = new();
    private readonly List<KernelRenderSpec> _renderKernels = new();
    private readonly SharedHelperRegistry _sharedHelperRegistry = new();
    private readonly PyNTTTargetOptions _targetOptions;

    public PyNTTKernelSourceConvertVisitor(CompileOptions compileOptions)
    {
        _targetOptions = PyNTTTargetOptionsUtility.Get(compileOptions);
    }

    public PyNTTGeneratedKernelSource GetKernelSource()
    {
        return new(_generatedKernels.ToArray(), _renderKernels.ToArray(), string.Empty);
    }

    protected override Unit VisitFunction(Function expr)
        => throw new NotSupportedException($"PyNTT kernel codegen expects lowered PrimFunction inputs, got Function {expr.Name}.");

    protected override Unit VisitFusion(Fusion expr)
        => throw new NotSupportedException($"PyNTT kernel codegen expects lowered PrimFunction inputs, got Fusion {expr.Name}.");

    protected override Unit VisitPrimFunctionWrapper(PrimFunctionWrapper expr)
        => throw new NotSupportedException($"PyNTT kernel codegen expects direct PrimFunction inputs, got PrimFunctionWrapper {expr.Name}.");

    protected override Unit VisitFunctionWrapper(FunctionWrapper expr)
        => throw new NotSupportedException($"PyNTT kernel codegen expects direct PrimFunction inputs, got FunctionWrapper {expr.Name}.");

    protected override Unit VisitPrimFunction(PrimFunction expr)
    {
        if (PyNTTPrimFunctionRoles.IsDispatchFunction(expr))
        {
            if (ContainsOwnExecutableKernelWork(expr.Body))
            {
                throw new NotSupportedException(
                    $"PyNTT Dispatch PrimFunction {expr.Name} contains executable kernel work. " +
                    "Dispatch functions must contain host-runtime control flow and calls to runtime-selected Compute functions only.");
            }

            if (ContainsDirectScheduledRegionCall(expr.Body))
            {
                throw new NotSupportedException(
                    $"PyNTT Dispatch PrimFunction {expr.Name} directly calls a ScheduledRegion function. " +
                    "A runtime branch must select a Compute top-kernel function, not a device-only region.");
            }

            return default;
        }

        // Auto-tiling device functions are excluded by PyNTTModuleBuilder and are
        // emitted through their runtime caller. A runtime PrimFunction therefore
        // owns the transitive work of its device callees even when its own body is
        // only a call wrapper.
        if (!ContainsTransitiveExecutableKernelWork(expr.Body))
        {
            return default;
        }

        var outputs = GetOutputInfos(expr);
        if (outputs.Length == 0)
        {
            throw new NotSupportedException($"PyNTT PrimFunction {expr.Name} contains kernel work but does not have caller-allocated tensor output storage.");
        }

        AddPrimFunctionKernel(expr, expr.Body, outputs);
        return default;
    }

    private void AddPrimFunctionKernel(PrimFunction function, BaseExpr body, OutputInfo[] outputs)
        => AddPrimFunctionKernel(BuildPrimFunctionKernel(function, body, outputs));

    private GeneratedPrimFunctionKernel BuildPrimFunctionKernel(PrimFunction function, BaseExpr body, OutputInfo[] outputs)
    {
        var parameterNames = function.Parameters.ToArray()
            .ToDictionary(parameter => parameter, parameter => parameter.Name);
        return new PyNTTPrimFunctionSourceVisitor(function, body, parameterNames, outputs, _targetOptions, _sharedHelperRegistry).Build();
    }

    private void AddPrimFunctionKernel(GeneratedPrimFunctionKernel kernel)
    {
        _generatedKernels.Add(kernel.Metadata);
        _renderKernels.Add(kernel.RenderSpec);
    }

    private static bool ContainsOwnExecutableKernelWork(BaseExpr expr)
        => ContainsExecutableKernelWork(expr, false, new HashSet<PrimFunction>(ReferenceEqualityComparer.Instance));

    private static bool ContainsTransitiveExecutableKernelWork(BaseExpr expr)
        => ContainsExecutableKernelWork(expr, true, new HashSet<PrimFunction>(ReferenceEqualityComparer.Instance));

    private static bool ContainsDirectScheduledRegionCall(BaseExpr expr)
    {
        if (expr is Call call)
        {
            if (call.Target is PrimFunction callee && PyNTTPrimFunctionRoles.IsScheduledRegionFunction(callee))
            {
                return true;
            }

            return call.Arguments.ToArray().Any(ContainsDirectScheduledRegionCall);
        }

        if (expr is BaseFunction)
        {
            return false;
        }

        foreach (var operand in expr.Operands)
        {
            if (ContainsDirectScheduledRegionCall(operand))
            {
                return true;
            }
        }

        return false;
    }

    private static bool ContainsExecutableKernelWork(BaseExpr expr, bool includeCallees, HashSet<PrimFunction> activeFunctions)
    {
        if (expr is Call call)
        {
            switch (call.Target)
            {
                case PrimFunction callee when includeCallees:
                    if (!activeFunctions.Add(callee))
                    {
                        throw new NotSupportedException($"PyNTT device call graph contains a recursive call involving {callee.Name}.");
                    }

                    try
                    {
                        return ContainsExecutableKernelWork(callee.Body, includeCallees, activeFunctions);
                    }
                    finally
                    {
                        activeFunctions.Remove(callee);
                    }

                case BaseFunction:
                    return false;
                case Nncase.TIR.NTT.Barrier or Nncase.TIR.NTT.SynchronizeThreads:
                    return true;
                case Op op when op.Parameters.Any(parameter => parameter.MemoryEffect is { Mode: not MemoryAccessMode.None }):
                    return true;
                default:
                    return call.Arguments.ToArray().Any(argument => ContainsExecutableKernelWork(argument, includeCallees, activeFunctions));
            }
        }

        if (expr is IfThenElse ifThenElse)
        {
            return ContainsExecutableKernelWork(ifThenElse.Then, includeCallees, activeFunctions) ||
                ContainsExecutableKernelWork(ifThenElse.Else, includeCallees, activeFunctions);
        }

        if (expr is Function or Fusion or FunctionWrapper or PrimFunctionWrapper)
        {
            throw new NotSupportedException($"PyNTT kernel codegen expects lowered PrimFunction bodies only, but found {expr.GetType().Name}. TIR selection and RemoveFunctionWrapperPass must run before PyNTT codegen.");
        }

        if (expr is BaseFunction)
        {
            return false;
        }

        foreach (var operand in expr.Operands)
        {
            if (ContainsExecutableKernelWork(operand, includeCallees, activeFunctions))
            {
                return true;
            }
        }

        return false;
    }

#pragma warning disable SA1201
    private sealed class PyNTTPrimFunctionSourceVisitor : ExprFunctor<Unit, Unit>
    {
        private enum HelperBarrierKind
        {
            None,
            Block,
            Grid,
        }

        private enum ReductionKernelKind
        {
            Matmul,
            PackedMatmul,
            QKVParallelLinear,
            PackedQKVParallelLinear,
            MatMulGlu,
            PackedMatMulGlu,
            Reduce,
        }

        private const string ShardCoordDimPrefix = "__shard_coord_";
        private const string DeviceFunctionCallPrefix = "__pyntt_device_call__";
        private const string PoolScopeSizeSuffix = "_pool_scope_size";

        private static readonly string[] WorkspaceParameterNames =
        {
            "data",
            "rdata",
            "chip_local_rdata",
            "chip_local_data",
            "block_local_rdata",
            "block_local_data",
            "data_pool_stride_bytes",
            "block_local_data_pool_stride_bytes",
        };

        private static string[] CollectLiveParameters(string bodySource, IEnumerable<string> candidates)
        {
            return candidates
                .Where(name => ContainsIdentifier(bodySource, name))
                .Distinct(StringComparer.Ordinal)
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();
        }

        private readonly PrimFunction _function;
        private readonly BaseExpr _bodyExpr;
        private readonly IReadOnlyDictionary<IVar, string> _parameterNames;
        private readonly OutputInfo[] _outputs;
        private readonly PyNTTTargetOptions _targetOptions;
        private readonly string _ownerName;
        private readonly bool _validateOutputs;
        private readonly StringBuilder _body = new();
        private readonly List<HelperTemplateRenderSpec> _helpers = new();
        private readonly List<DeviceFunctionRenderSpec> _deviceFunctions = new();
        private readonly List<string> _inputNames;
        private readonly List<string> _opKinds = new();
        private readonly List<HelperKernelCallMetadata> _helperCalls = new();
        private readonly Dictionary<string, string[]> _helperArguments = new(StringComparer.Ordinal);
        private readonly Dictionary<string, string[]> _helperScalarArguments = new(StringComparer.Ordinal);
        private readonly List<PyNTTKVCacheFieldInputMetadata> _kvCacheFieldInputs;
        private readonly Dictionary<string, PyNTTKVCacheStorageMetadata?> _formalObjectFieldStorages = new(StringComparer.Ordinal);
        private readonly SortedSet<string> _runtimeScalarNames;
        private readonly SortedSet<string> _helperScalarNameCandidates = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _activeLocalScalarNames = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _activeLocalBufferNames = new(StringComparer.Ordinal);
        private readonly Dictionary<string, Stack<PyNTTDimExpression>> _activeLoopVariableRanges = new(StringComparer.Ordinal);
        private readonly SortedSet<string> _abiViewStrideArgNames;
        private readonly Dictionary<string, int> _helperCounters = new();
        private readonly Dictionary<string, int> _primFunctionCallCounters = new(StringComparer.Ordinal);
        private readonly Stack<string> _semanticHelperScopes = new();
        private readonly Dictionary<string, object> _attrs = new();
        private readonly Dictionary<TIR.Buffer, int> _bufferInputIndices;
        private readonly Dictionary<BufferVar, TIR.Buffer> _abiBufferMemo;
        private readonly Dictionary<TIR.Buffer, string> _dataBaseNameByBuffer;
        private readonly Dictionary<TIR.Buffer, string> _chipLocalDataBaseNameByBuffer;
        private readonly Dictionary<TIR.Buffer, string> _blockLocalDataBaseNameByBuffer;
        private readonly Dictionary<TIR.Buffer, PyNTTDimExpression[]> _bufferActiveShapeOverrides = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<TIR.Buffer, PyNTTDimExpression[]> _bufferGlobalShapeOverrides = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<TIR.Buffer, PyNTTDimExpression[]> _bufferGlobalOffsetOverrides = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<TIR.Buffer, int[][]> _bufferSourceSplitAxesOverrides = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<TIR.Buffer, BufferViewSource> _bufferViewSourceByBuffer = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<TIR.Buffer, SharedBufferAllocation> _sharedBufferAllocations = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<IVar, BaseExpr> _letBindings = new(ReferenceEqualityComparer.Instance);
        private readonly IReadOnlyDictionary<IVar, string> _formalTensorParameterBaseNames;
        private readonly IReadOnlyDictionary<IVar, string> _formalTensorParameterPoolStrideNames;
        private readonly IReadOnlyDictionary<IVar, string> _formalTensorParameterPoolScopeSizeNames;
        private readonly IReadOnlyDictionary<IVar, PyNTTDimExpression[]> _formalTensorParameterDimensions;
        private readonly IReadOnlyDictionary<IVar, PyNTTDimExpression[]> _formalTensorParameterGlobalOffsets;
        private readonly IReadOnlyDictionary<IVar, int[][]> _formalTensorParameterSourceSplitAxes;
        private readonly IReadOnlyDictionary<string, string> _formalDimParameterNames;
        private readonly IReadOnlyDictionary<IVar, string> _formalObjectParameterBaseNames;
        private readonly Dictionary<TIR.Buffer, TIR.Buffer> _objectViewSourceByBuffer = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<TIR.Buffer, string> _formalObjectBaseNameByBuffer = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<int, string> _formalObjectOutputAliases = new();
        private readonly HashSet<IVar> _formalWorkspaceParameters;
        private readonly SortedSet<string> _extraWorkspaceBaseNames;
        private readonly Dictionary<string, string> _extraPointerParameterTritonTypes;
        private readonly string _dataBaseName;
        private readonly string _chipLocalDataBaseName;
        private readonly string _blockLocalDataBaseName;
        private readonly HashSet<int> _storedOutputIndices;
        private readonly HashSet<int> _definitelyStoredOutputIndices;
        private readonly DistributedType?[] _outputDistributedTypes;
        private readonly Dictionary<int, int> _outputAliases;
        private readonly SharedHelperRegistry _sharedHelperRegistry;
        private readonly PyNTTDimExpressionEmitter _dimEmitter;
        private readonly HashSet<PrimFunction> _activePrimFunctionCalls;
        private readonly HashSet<string> _activeDeviceFunctionNames;
        private readonly Dictionary<PrimFunction, DeviceFunctionDefinition> _deviceFunctionDefinitions;
        private readonly Dictionary<string, DeviceFunctionDefinition> _deviceFunctionDefinitionsByName;
        private readonly Stack<Dictionary<Call, ReductionState>> _reductionScopes = new();
        private readonly List<Dictionary<string, object>> _blockMicroKernelContracts = new();
        private readonly HashSet<string> _blockMicroKernelContractKeys = new(StringComparer.Ordinal);
        private long _nestedBlockLocalDataPoolBytes;
        private long _nestedSharedMemoryBytes;
        private long _sharedAliasRequiredBytes;
        private readonly PrimFunction _currentFunction;
        private int _bodyIndent;
        private int _reductionStateCounter;
        private int _sharedBufferAllocationCounter;
        private ReductionState? _currentReductionState;
        private BlockMicroKernelSelection? _currentBlockMicroKernel;

        public PyNTTPrimFunctionSourceVisitor(
            PrimFunction function,
            BaseExpr bodyExpr,
            IReadOnlyDictionary<IVar, string> parameterNames,
            OutputInfo[] outputs,
            PyNTTTargetOptions targetOptions,
            SharedHelperRegistry sharedHelperRegistry,
            KernelAbiState? abiState = null,
            string? ownerName = null,
            bool validateOutputs = true,
            PrimFunction? currentFunction = null,
            Dictionary<TIR.Buffer, string>? dataBaseNameByBuffer = null,
            Dictionary<TIR.Buffer, string>? chipLocalDataBaseNameByBuffer = null,
            Dictionary<TIR.Buffer, string>? blockLocalDataBaseNameByBuffer = null,
            IReadOnlyDictionary<IVar, string>? formalTensorParameterBaseNames = null,
            IReadOnlyDictionary<IVar, string>? formalTensorParameterPoolStrideNames = null,
            IReadOnlyDictionary<IVar, string>? formalTensorParameterPoolScopeSizeNames = null,
            IReadOnlyDictionary<IVar, PyNTTDimExpression[]>? formalTensorParameterDimensions = null,
            IReadOnlyDictionary<IVar, PyNTTDimExpression[]>? formalTensorParameterGlobalOffsets = null,
            IReadOnlyDictionary<IVar, int[][]>? formalTensorParameterSourceSplitAxes = null,
            IReadOnlyDictionary<string, string>? formalDimParameterNames = null,
            IReadOnlyDictionary<IVar, string>? formalObjectParameterBaseNames = null,
            IEnumerable<IVar>? formalWorkspaceParameters = null,
            IEnumerable<string>? extraWorkspaceBaseNames = null,
            IReadOnlyDictionary<string, string>? extraPointerParameterTritonTypes = null,
            string dataBaseName = "data",
            string chipLocalDataBaseName = "chip_local_data",
            string blockLocalDataBaseName = "block_local_data")
        {
            _function = function;
            _currentFunction = currentFunction ?? function;
            _bodyExpr = bodyExpr;
            _parameterNames = parameterNames;
            _outputs = outputs;
            _targetOptions = targetOptions;
            _ownerName = ownerName ?? function.Name;
            _validateOutputs = validateOutputs;
            _sharedHelperRegistry = sharedHelperRegistry;
            abiState ??= new KernelAbiState(outputs.Length);
            _inputNames = abiState.InputNames;
            _kvCacheFieldInputs = abiState.KVCacheFieldInputs;
            _runtimeScalarNames = abiState.RuntimeScalarNames;
            _abiViewStrideArgNames = abiState.AbiViewStrideArgNames;
            _bufferInputIndices = abiState.BufferInputIndices;
            _abiBufferMemo = abiState.AbiBufferMemo;
            _dataBaseNameByBuffer = dataBaseNameByBuffer ?? new Dictionary<TIR.Buffer, string>(ReferenceEqualityComparer.Instance);
            _chipLocalDataBaseNameByBuffer = chipLocalDataBaseNameByBuffer ?? new Dictionary<TIR.Buffer, string>(ReferenceEqualityComparer.Instance);
            _blockLocalDataBaseNameByBuffer = blockLocalDataBaseNameByBuffer ?? new Dictionary<TIR.Buffer, string>(ReferenceEqualityComparer.Instance);
            _formalTensorParameterBaseNames = formalTensorParameterBaseNames ?? new Dictionary<IVar, string>(ReferenceEqualityComparer.Instance);
            _formalTensorParameterPoolStrideNames = formalTensorParameterPoolStrideNames ?? new Dictionary<IVar, string>(ReferenceEqualityComparer.Instance);
            _formalTensorParameterPoolScopeSizeNames = formalTensorParameterPoolScopeSizeNames ?? new Dictionary<IVar, string>(ReferenceEqualityComparer.Instance);
            _formalTensorParameterDimensions = formalTensorParameterDimensions ?? new Dictionary<IVar, PyNTTDimExpression[]>(ReferenceEqualityComparer.Instance);
            _formalTensorParameterGlobalOffsets = formalTensorParameterGlobalOffsets ?? new Dictionary<IVar, PyNTTDimExpression[]>(ReferenceEqualityComparer.Instance);
            _formalTensorParameterSourceSplitAxes = formalTensorParameterSourceSplitAxes ?? new Dictionary<IVar, int[][]>(ReferenceEqualityComparer.Instance);
            _formalDimParameterNames = formalDimParameterNames ?? new Dictionary<string, string>(StringComparer.Ordinal);
            _formalObjectParameterBaseNames = formalObjectParameterBaseNames ?? new Dictionary<IVar, string>(ReferenceEqualityComparer.Instance);
            _formalWorkspaceParameters = formalWorkspaceParameters is null ? new HashSet<IVar>(ReferenceEqualityComparer.Instance) : new HashSet<IVar>(formalWorkspaceParameters, ReferenceEqualityComparer.Instance);
            _extraWorkspaceBaseNames = extraWorkspaceBaseNames is null ? new SortedSet<string>(StringComparer.Ordinal) : new SortedSet<string>(extraWorkspaceBaseNames, StringComparer.Ordinal);
            _extraPointerParameterTritonTypes = extraPointerParameterTritonTypes is null
                ? new Dictionary<string, string>(StringComparer.Ordinal)
                : new Dictionary<string, string>(extraPointerParameterTritonTypes, StringComparer.Ordinal);
            _dataBaseName = dataBaseName;
            _chipLocalDataBaseName = chipLocalDataBaseName;
            _blockLocalDataBaseName = blockLocalDataBaseName;
            _storedOutputIndices = abiState.StoredOutputIndices;
            _definitelyStoredOutputIndices = new HashSet<int>(_storedOutputIndices);
            _outputDistributedTypes = abiState.OutputDistributedTypes;
            _outputAliases = abiState.OutputAliases;
            _activePrimFunctionCalls = abiState.ActivePrimFunctionCalls;
            _activeDeviceFunctionNames = abiState.ActiveDeviceFunctionNames;
            _deviceFunctionDefinitions = abiState.DeviceFunctionDefinitions;
            _deviceFunctionDefinitionsByName = abiState.DeviceFunctionDefinitionsByName;
            _dimEmitter = new(RegisterRuntimeScalar, FormatRuntimeScalar, BuildThreadIdExpression(targetOptions), ResolveActiveLoopVariable);
        }

        public GeneratedPrimFunctionKernel Build()
        {
            _attrs["tir"] = true;
            Visit(_bodyExpr);
            RegisterInOutObjectOutputAliases();
            var bodySource = _body.ToString().TrimEnd();
            var inputLayout = BuildKernelInputLayout(bodySource, _deviceFunctions);
            var materializedOutputIndices = _definitelyStoredOutputIndices.Concat(_outputAliases.Keys).ToHashSet();
            if (_validateOutputs && materializedOutputIndices.Count != _outputs.Length)
            {
                var missingOutputs = _outputs
                    .Select((output, index) => (output.Name, Index: index))
                    .Where(output => !materializedOutputIndices.Contains(output.Index))
                    .Select(output => output.Name);
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} does not materialize output(s): {string.Join(", ", missingOutputs)}.");
            }

            var outputs = _outputs
                .Select((output, index) => output with
                {
                    DistributedType = _outputDistributedTypes[index] ?? output.DistributedType,
                })
                .ToArray();
            var opKind = _opKinds.Count == 1 ? _opKinds[0] : "composite";
            if (_opKinds.Count == 0)
            {
                opKind = _body.Length == 0 ? "alias" : "copy";
            }

            if (_opKinds.Count > 1)
            {
                _attrs["ops"] = _opKinds.ToArray();
            }

            var kernelOutputIndexes = outputs
                .Select((output, index) => (output, index))
                .Where(item => materializedOutputIndices.Contains(item.index) && !IsObjectDataType(item.output.DType))
                .ToArray();
            var kernelOutputs = kernelOutputIndexes.Select(item => item.output).ToArray();
            var kernelOutputIndexByFunctionIndex = kernelOutputIndexes
                .Select((item, kernelIndex) => (item.index, kernelIndex))
                .ToDictionary(item => item.index, item => item.kernelIndex);
            var kernelOutputAliases = new Dictionary<int, int>();
            var runtimeOutputAliases = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var pair in _outputAliases)
            {
                var sourceInputName = GetInputName(pair.Value, $"PyNTT output alias {outputs[pair.Key].Name}");
                runtimeOutputAliases[outputs[pair.Key].Name] = sourceInputName;
                if (kernelOutputIndexByFunctionIndex.TryGetValue(pair.Key, out var kernelOutputIndex))
                {
                    if (!inputLayout.IndexMap.TryGetValue(pair.Value, out var kernelInputIndex))
                    {
                        throw new NotSupportedException($"PyNTT output {outputs[pair.Key].Name} aliases object input {sourceInputName}, which cannot be passed to a Triton kernel.");
                    }

                    kernelOutputAliases[kernelOutputIndex] = kernelInputIndex;
                }
            }

            if (kernelOutputAliases.Count > 0)
            {
                _attrs["output_aliases"] = kernelOutputAliases;
            }

            if (runtimeOutputAliases.Count > 0)
            {
                _attrs["runtime_output_aliases"] = runtimeOutputAliases;
            }

            if (_runtimeScalarNames.Count > 0)
            {
                _attrs["runtime_shape_args"] = _runtimeScalarNames.ToArray();
            }

            if (_abiViewStrideArgNames.Count > 0)
            {
                _attrs["abi_view_stride_args"] = _abiViewStrideArgNames
                    .Select(argument => RemapInputReferences(argument, inputLayout.IndexMap, inputLayout.RemovedIndexes, $"PyNTT PrimFunction {_function.Name} ABI view stride argument {argument}"))
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(argument => argument.StartsWith("input", StringComparison.Ordinal) ? 0 : 1)
                    .ThenBy(ParseAbiArgumentIndex)
                    .ThenBy(argument => argument, StringComparer.Ordinal)
                    .ToArray();
            }

            if (_kvCacheFieldInputs.Count > 0)
            {
                _attrs["kv_cache_field_inputs"] = _kvCacheFieldInputs.ToArray();
            }

            if (_helperCalls.Count > 0)
            {
                _attrs["helper_calls"] = _helperCalls.ToArray();
            }

            if (opKind == "alias")
            {
                _attrs["pure_alias"] = true;
            }

            var sharedMemoryBytes = GetSharedMemoryBytes();
            if (sharedMemoryBytes > 0)
            {
                _attrs["shared_memory_bytes"] = sharedMemoryBytes;
            }

            AddTargetResourceMetadata();

            var metadata = new GeneratedKernelMetadata(
                SanitizePythonIdentifier($"{_function.Name}_{opKind}_0"),
                opKind,
                inputLayout.Names,
                kernelOutputs.Select(output => output.Name).ToArray(),
                _attrs,
                BuildLaunchMetadata(
                    kernelOutputs.Length > 0 ? kernelOutputs[0] : outputs[0],
                    _targetOptions,
                    new()
                    {
                        ["data_pool_bytes"] = checked((long)_currentFunction.SchedResult.DataUsage),
                        ["data_pool_elements"] = checked((long)_currentFunction.SchedResult.DataUsage),
                        ["data_dtype"] = "uint8",
                        ["chip_local_data_pool_bytes"] = checked((long)_currentFunction.SchedResult.ChipLocalDataPoolSize),
                        ["block_local_data_pool_bytes"] = GetBlockLocalDataPoolBytes(),
                        ["block_local_data_scope_count"] = GetBlockLocalDataScopeCount(_targetOptions),
                        ["rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.Rdatas),
                        ["chip_local_rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.ChipLocalRdatas),
                        ["block_local_rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.BlockLocalRdatas),
                        ["block_local_rdata_stride_bytes"] = PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_function.SchedResult.BlockLocalRdatas, _targetOptions, "b"),
                    }));
            var renderSpec = new KernelRenderSpec(
                metadata,
                inputLayout.Helpers,
                inputLayout.DeviceFunctions,
                inputLayout.BodySource);
            return new(metadata, renderSpec);
        }

        private DeviceFunctionBuildResult BuildDeviceFunction(
            string name,
            IReadOnlyDictionary<string, string> parameterOverrides,
            IReadOnlyDictionary<string, string> extraParameterArguments)
        {
            Visit(_bodyExpr);
            RegisterInOutObjectOutputAliases();
            var isScheduledRegionFunction = PyNTTPrimFunctionRoles.IsScheduledRegionFunction(_currentFunction);
            var bodySource = _body.ToString().TrimEnd();
            var liveExtraParameters = CollectLiveParameters(bodySource, _extraWorkspaceBaseNames);
            return new(
                new DeviceFunctionRenderSpec(
                    name,
                    NoInline: true,
                    PreserveHelperCallBoundaries: _currentFunction.Role == FunctionRole.Compute && !isScheduledRegionFunction,
                    _helpers.ToArray(),
                    bodySource,
                    parameterOverrides,
                    liveExtraParameters,
                    extraParameterArguments),
                _deviceFunctions.ToArray(),
                _helperCalls.ToArray(),
                _opKinds.ToArray(),
                _attrs.TryGetValue("requires_grid_barrier", out var requiresGridBarrier) && requiresGridBarrier is true,
                GetBlockLocalDataPoolBytes(),
                GetSharedMemoryBytes(),
                new Dictionary<string, PyNTTKVCacheStorageMetadata?>(_formalObjectFieldStorages, StringComparer.Ordinal),
                new Dictionary<int, string>(_formalObjectOutputAliases));
        }

        private void RegisterInOutObjectOutputAliases()
        {
            var outputs = _currentFunction.GetAbiView().OutputParameters;
            for (var outputIndex = 0; outputIndex < outputs.Count; outputIndex++)
            {
                var output = outputs[outputIndex];
                if (output.Role != BufferVarRole.InOut || !IsObjectDataType(output.CheckedDataType))
                {
                    continue;
                }

                if (TryGetFormalObjectBaseName(output, out var formalObjectBaseName))
                {
                    RecordFormalObjectOutputAlias(outputIndex, formalObjectBaseName, "InOut ABI");
                }
                else if (TryResolveObjectInputIndex(output, out var inputIndex))
                {
                    RecordRuntimeObjectOutputAlias(outputIndex, inputIndex, "InOut ABI");
                }
                else
                {
                    throw new NotSupportedException($"PyNTT PrimFunction {_currentFunction.Name} InOut object {output.Name} cannot be resolved to an input ABI parameter.");
                }
            }
        }

        protected override Unit VisitTuple(Nncase.IR.Tuple expr)
        {
            foreach (var field in expr.Fields)
            {
                Visit(field);
            }

            return default;
        }

        protected override Unit VisitSequential(Sequential expr)
        {
            var semanticScopeName = expr.TraceScopeName;
            if (semanticScopeName is not null)
            {
                _semanticHelperScopes.Push(semanticScopeName);
            }

            try
            {
                if (expr.PreserveCodegenBoundary)
                {
                    if (semanticScopeName is not { } codegenScopeName)
                    {
                        throw new InvalidOperationException(
                            $"PyNTT codegen scope in {_currentFunction.Name} must have a semantic trace name.");
                    }

                    return VisitCodegenScope(expr, codegenScopeName);
                }

                var traceLabel = semanticScopeName is not null && !ReferenceEquals(expr, _currentFunction.Body)
                    ? GetPrimFunctionCallTraceLabel(semanticScopeName)
                    : null;
                if (traceLabel is not null)
                {
                    WriteTraceMarker($"begin_function:{traceLabel}");
                }

                VisitSequentialFields(expr);

                if (traceLabel is not null)
                {
                    WriteTraceMarker($"end_function:{traceLabel}");
                }

                return default;
            }
            finally
            {
                if (semanticScopeName is not null)
                {
                    _semanticHelperScopes.Pop();
                }
            }
        }

        private Unit VisitCodegenScope(Sequential expr, string scopeName)
        {
            if (_reductionScopes.Count > 0 || _currentReductionState is not null)
            {
                throw new NotSupportedException(
                    $"PyNTT codegen scope {scopeName} in {_currentFunction.Name} crosses an active reduction boundary. " +
                    "A codegen scope must own the complete reduction so its accumulator state remains local.");
            }

            var outerLocalNames = _activeLocalScalarNames.Keys
                .Concat(_activeLocalBufferNames.Keys)
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();
            var bodyStart = _body.Length;
            var scopeIndent = _bodyIndent;
            VisitSequentialFields(expr);
            var capturedBody = _body.ToString(bodyStart, _body.Length - bodyStart);
            _body.Remove(bodyStart, _body.Length - bodyStart);

            var bodySource = DedentGeneratedBody(capturedBody, scopeIndent);
            var traceLabel = GetPrimFunctionCallTraceLabel(scopeName);
            WriteTraceMarker($"begin_function:{traceLabel}");
            if (!string.IsNullOrWhiteSpace(bodySource))
            {
                var deviceFunctionName = _sharedHelperRegistry.GetNextCodegenScopeDeviceFunctionName(
                    _ownerName,
                    scopeName);
                var extraParameters = CollectLiveParameters(
                    bodySource,
                    outerLocalNames.Concat(_extraWorkspaceBaseNames));
                _deviceFunctions.Add(
                    new DeviceFunctionRenderSpec(
                        deviceFunctionName,
                        NoInline: true,
                        PreserveHelperCallBoundaries: false,
                        Array.Empty<HelperTemplateRenderSpec>(),
                        bodySource,
                        new Dictionary<string, string>(StringComparer.Ordinal),
                        extraParameters,
                        new Dictionary<string, string>(StringComparer.Ordinal)));
                WriteControlLine(BuildDeviceFunctionCallPlaceholder(deviceFunctionName, extraParameters));
            }

            WriteTraceMarker($"end_function:{traceLabel}");
            return default;
        }

        private void VisitSequentialFields(Sequential expr)
        {
            for (var index = 0; index < expr.Fields.Length; index++)
            {
                var field = expr.Fields[index];
                if (field is Function or Fusion or FunctionWrapper or PrimFunctionWrapper)
                {
                    throw new NotSupportedException($"PyNTT kernel codegen expects lowered PrimFunction bodies only, but found {field.GetType().Name}. TIR selection and RemoveFunctionWrapperPass must run before PyNTT codegen.");
                }

                if (field is BaseFunction)
                {
                    continue;
                }

                if (ReductionCodegenUtility.TryGetAdjacentReductionLoopPartitionPair(expr.Fields, index, out var fullLoop, out var tailLoop))
                {
                    if (_reductionScopes.Count == 0)
                    {
                        VisitPartitionedReductionLoops(fullLoop, tailLoop);
                    }
                    else
                    {
                        Visit(fullLoop);
                        Visit(tailLoop);
                    }

                    index++;
                    continue;
                }

                Visit(field);
            }
        }

        private static string DedentGeneratedBody(string source, int indent)
        {
            var prefix = new string(' ', checked(indent * 4));
            var lines = source.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');
            for (var index = 0; index < lines.Length; index++)
            {
                if (string.IsNullOrWhiteSpace(lines[index]) || prefix.Length == 0)
                {
                    continue;
                }

                if (!lines[index].StartsWith(prefix, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException(
                        $"PyNTT generated code escaped its enclosing indentation while outlining a codegen scope: {lines[index]}");
                }

                lines[index] = lines[index][prefix.Length..];
            }

            return string.Join('\n', lines).TrimEnd();
        }

        protected override Unit VisitBuffer(TIR.Buffer expr)
        {
            return default;
        }

        protected override Unit VisitIfThenElse(IfThenElse expr)
        {
            var incomingOutputState = CaptureOutputControlFlowState();
            WriteControlLine($"if {BuildScalarExpression(expr.Condition)}:");
            _bodyIndent++;
            Visit(expr.Then);
            _bodyIndent--;
            var thenOutputState = CaptureOutputControlFlowState();

            RestoreOutputControlFlowState(incomingOutputState);

            if (expr.Else.Count > 0)
            {
                WriteControlLine("else:");
                _bodyIndent++;
                Visit(expr.Else);
                _bodyIndent--;
            }

            var elseOutputState = CaptureOutputControlFlowState();
            MergeConditionalOutputStates(thenOutputState, elseOutputState);
            return default;
        }

        protected override Unit VisitFor(For expr)
        {
            if (expr.Mode is not (LoopMode.Serial or LoopMode.Reduction))
            {
                throw new NotSupportedException($"PyNTT codegen only supports serial and reduction TIR For loops, got {expr.Mode}.");
            }

            var ownsReductionScope = expr.Mode == LoopMode.Reduction && _reductionScopes.Count == 0;
            Dictionary<Call, ReductionState>? reductionScope = null;
            var loopStart = _body.Length;
            if (ownsReductionScope)
            {
                reductionScope = CreateReductionScope(expr);
                _reductionScopes.Push(reductionScope);
            }

            var loopVar = SanitizePythonIdentifier(expr.LoopVar.Name);
            var start = GetDimensionExpression(expr.Domain.Start);
            var stop = GetDimensionExpression(expr.Domain.Stop);
            var step = GetDimensionExpression(expr.Domain.Step);
            WriteControlLine($"for {loopVar} in tl.range({start.TritonExpression}, {stop.TritonExpression}, {step.TritonExpression}):");
            PushLocalScalar(loopVar);
            PushLoopVariableRange(loopVar, GetLoopVariableRange(loopVar, start, stop, step));
            _bodyIndent++;
            try
            {
                Visit(expr.Body);
            }
            finally
            {
                _bodyIndent--;
                PopLoopVariableRange(loopVar);
                PopLocalScalar(loopVar);
            }

            if (ownsReductionScope)
            {
                try
                {
                    var initSource = BuildReductionInitializers(reductionScope!, _bodyIndent);
                    _body.Insert(loopStart, initSource);
                    foreach (var state in GetDistinctReductionStates(reductionScope!))
                    {
                        EmitReductionFinalize(state);
                    }
                }
                finally
                {
                    _reductionScopes.Pop();
                }
            }

            return default;
        }

        private void VisitPartitionedReductionLoops(For fullLoop, For tailLoop)
        {
            var reductionScope = CreateReductionScope(fullLoop.Body, tailLoop.Body);
            var loopStart = _body.Length;
            _reductionScopes.Push(reductionScope);
            try
            {
                Visit(fullLoop);
                Visit(tailLoop);
                _body.Insert(loopStart, BuildReductionInitializers(reductionScope, _bodyIndent));
                foreach (var state in GetDistinctReductionStates(reductionScope))
                {
                    EmitReductionFinalize(state);
                }
            }
            finally
            {
                _reductionScopes.Pop();
            }
        }

        protected override Unit VisitLet(Let expr)
        {
            if (expr.Var is DimVar dimVar)
            {
                var localName = SanitizePythonIdentifier(dimVar.Name);
                var scalarValue = BuildScalarExpression(expr.Expression);
                WriteControlLine($"{localName} = {scalarValue}");
                var hadPreviousDimBinding = _letBindings.TryGetValue(expr.Var, out var previousDimBinding);
                _letBindings[expr.Var] = expr.Expression;
                PushLocalScalar(localName);

                try
                {
                    Visit(expr.Body);
                }
                finally
                {
                    PopLocalScalar(localName);
                    if (hadPreviousDimBinding)
                    {
                        _letBindings[expr.Var] = previousDimBinding!;
                    }
                    else
                    {
                        _letBindings.Remove(expr.Var);
                    }
                }

                return default;
            }

            var boundExpression = UnwrapInputBoxing(expr.Expression);
            var value = MaterializeLetBinding(expr.Var, boundExpression);
            var hadPrevious = _letBindings.TryGetValue(expr.Var, out var previous);
            _letBindings[expr.Var] = value;
            SharedBufferAllocation? sharedAllocation = null;
            var hadPreviousSharedAllocation = false;
            SharedBufferAllocation? previousSharedAllocation = null;
            if (boundExpression is Call allocateBufferView &&
                allocateBufferView.Target is Nncase.IR.Buffers.AllocateBufferView &&
                value is TIR.Buffer sharedBuffer &&
                sharedBuffer.MemSpan.Buffer.Location == MemoryLocation.Shared)
            {
                sharedAllocation = EmitSharedBufferAllocation(expr.Var, sharedBuffer);
                hadPreviousSharedAllocation = _sharedBufferAllocations.TryGetValue(sharedBuffer, out previousSharedAllocation);
                _sharedBufferAllocations[sharedBuffer] = sharedAllocation;
                PushLocalBuffer(sharedAllocation.DescriptorName);
            }

            try
            {
                Visit(expr.Body);
            }
            finally
            {
                if (sharedAllocation is not null)
                {
                    PopLocalBuffer(sharedAllocation.DescriptorName);
                    if (hadPreviousSharedAllocation)
                    {
                        _sharedBufferAllocations[(TIR.Buffer)value] = previousSharedAllocation!;
                    }
                    else
                    {
                        _sharedBufferAllocations.Remove((TIR.Buffer)value);
                    }
                }

                if (hadPrevious)
                {
                    _letBindings[expr.Var] = previous!;
                }
                else
                {
                    _letBindings.Remove(expr.Var);
                }
            }

            return default;
        }

        protected override Unit VisitReturn(Return expr)
        {
            return default;
        }

        protected override Unit VisitCall(Call expr)
        {
            var args = expr.Arguments.ToArray()
                .Select((arg, index) => ResolveCallArgument(arg, index))
                .ToArray();
            var previousReductionState = _currentReductionState;
            var previousBlockMicroKernel = _currentBlockMicroKernel;
            _currentReductionState = TryGetReductionState(expr, out var reductionState) ? reductionState : null;
            _currentBlockMicroKernel = expr.Metadata.BlockMicroKernel;
            try
            {
                switch (expr.Target)
                {
                    case Nncase.TIR.NTT.TensorLoad:
                        VisitTensorLoad(args);
                        break;
                    case Nncase.TIR.NTT.TensorStore:
                        VisitTensorStore(args);
                        break;
                    case Nncase.TIR.Memcopy:
                        VisitMemcopy(args);
                        break;
                    case Nncase.TIR.TileLoad:
                        VisitTileLoad(args);
                        break;
                    case Nncase.TIR.TileStore:
                        VisitTileStore(args);
                        break;
                    case Nncase.TIR.NTT.Unary unary:
                        VisitUnary(unary, args);
                        break;
                    case Nncase.TIR.NTT.Erf:
                        VisitElementwiseUnary(UnaryOp.Erf, "PyNTT Erf", args);
                        break;
                    case Nncase.TIR.NTT.Expand:
                        VisitExpand(args);
                        break;
                    case Nncase.TIR.NTT.Gather gather:
                        VisitGather(gather, args);
                        break;
                    case Nncase.TIR.NTT.GatherReduceScatter gatherReduceScatter:
                        VisitGatherReduceScatter(gatherReduceScatter, args);
                        break;
                    case Nncase.TIR.NTT.Pad pad:
                        VisitPad(pad, args);
                        break;
                    case Nncase.TIR.NTT.ScatterND:
                        VisitScatterND(args);
                        break;
                    case Nncase.TIR.NTT.Slice slice:
                        VisitSlice(slice, args);
                        break;
                    case Nncase.TIR.NTT.Swish swish:
                        VisitSwish(swish, args);
                        break;
                    case Nncase.TIR.NTT.VectorizedBinary binary:
                        VisitVectorizedBinary(binary, args);
                        break;
                    case Nncase.TIR.NTT.Pack pack:
                        VisitPack(pack, args);
                        break;
                    case Nncase.TIR.NTT.Unpack unpack:
                        VisitUnpack(unpack, args);
                        break;
                    case Nncase.TIR.NTT.Cast cast:
                        VisitCast(cast, args);
                        break;
                    case Nncase.TIR.NTT.Where:
                        VisitWhere(args);
                        break;
                    case Nncase.TIR.NTT.Clamp clamp:
                        VisitClamp(clamp, args);
                        break;
                    case Nncase.TIR.NTT.Compare compare:
                        VisitCompare(compare, args);
                        break;
                    case Nncase.TIR.NTT.Concat concat:
                        VisitConcat(concat, args);
                        break;
                    case Nncase.TIR.NTT.Conv2D conv2D:
                        VisitConv2D(conv2D, args);
                        break;
                    case Nncase.TIR.NTT.Transpose transpose:
                        VisitTranspose(transpose, args);
                        break;
                    case Nncase.TIR.NTT.Matmul matmul:
                        VisitMatmul(matmul, args);
                        break;
                    case Nncase.TIR.NTT.PackedMatMul packedMatmul:
                        VisitPackedMatmul(packedMatmul, args);
                        break;
                    case Nncase.TIR.NTT.QKVParallelLinear qkvParallelLinear:
                        VisitQKVParallelLinear(qkvParallelLinear, args);
                        break;
                    case Nncase.TIR.NTT.PackedQKVParallelLinear packedQKVParallelLinear:
                        VisitPackedQKVParallelLinear(packedQKVParallelLinear, args);
                        break;
                    case Nncase.TIR.NTT.MatMulGlu matMulGlu:
                        VisitMatMulGlu(matMulGlu, args);
                        break;
                    case Nncase.TIR.NTT.PackedMatMulGlu packedMatMulGlu:
                        VisitPackedMatMulGlu(packedMatMulGlu, args);
                        break;
                    case Nncase.TIR.NTT.SUMMA summa:
                        VisitSUMMA(summa, args);
                        break;
                    case Nncase.TIR.NTT.Reduce reduce:
                        VisitReduce(reduce, args);
                        break;
                    case Nncase.TIR.NTT.RoPE:
                        VisitRoPE(args);
                        break;
                    case Nncase.TIR.NTT.GetPositionIds getPositionIds:
                        VisitGetPositionIds(getPositionIds, args);
                        break;
                    case Nncase.TIR.NTT.UpdatePagedAttentionKVCache updatePagedAttentionKVCache:
                        VisitUpdatePagedAttentionKVCache(updatePagedAttentionKVCache, args);
                        break;
                    case Nncase.TIR.NTT.PagedAttention pagedAttention:
                        VisitPagedAttention(pagedAttention, args);
                        break;
                    case Nncase.TIR.NTT.VectorizedLayerNorm layerNorm:
                        VisitLayerNorm(layerNorm, args);
                        break;
                    case Nncase.TIR.NTT.NormStats normStats:
                        VisitNormStats(normStats, args);
                        break;
                    case Nncase.TIR.NTT.NormApply normApply:
                        VisitNormApply(normApply, args);
                        break;
                    case Nncase.TIR.NTT.SynchronizeThreads:
                        _attrs["requires_grid_barrier"] = true;
                        WriteBarrier(HelperBarrierKind.Grid);
                        break;
                    case Nncase.TIR.NTT.Barrier barrier:
                        WriteExplicitBarrier(barrier.Scope);
                        break;
                    case Nncase.TIR.NTT.VectorizedSoftmax softmax:
                        VisitSoftmax(softmax.Axis, softmax.VectorizedAxes, args, "softmax");
                        break;
                    case Nncase.TIR.NTT.Softmax softmax:
                        VisitSoftmax(softmax.Axis, default, args, "softmax");
                        break;
                    case PrimFunction callee:
                        VisitPrimFunctionCall(callee, args);
                        break;
                    case BaseFunction callee:
                        throw new NotSupportedException($"PyNTT kernel codegen expects direct PrimFunction call targets, got {callee.GetType().Name} {callee.Name}.");
                    default:
                        throw new NotSupportedException($"Unsupported PyNTT PrimFunction call target: {expr.Target.GetType().Name}.");
                }
            }
            finally
            {
                _currentReductionState = previousReductionState;
                _currentBlockMicroKernel = previousBlockMicroKernel;
            }

            return default;
        }

        private void VisitPrimFunctionCall(PrimFunction callee, IReadOnlyList<BaseExpr> args)
        {
            if (PyNTTPrimFunctionRoles.IsDispatchFunction(callee))
            {
                throw new NotSupportedException(
                    $"PyNTT Compute PrimFunction {_currentFunction.Name} calls Dispatch PrimFunction {callee.Name}. " +
                    "Runtime dispatch must be resolved by model.py before entering a Triton top kernel.");
            }

            var parameters = callee.Parameters.ToArray();
            if (parameters.Length != args.Count)
            {
                throw new NotSupportedException($"PyNTT call to {callee.Name} expects {parameters.Length} arguments, got {args.Count}.");
            }

            if (!ContainsTransitiveExecutableKernelWork(callee.Body))
            {
                throw new InvalidOperationException(
                    $"Descriptor-only PrimFunction {callee.Name} survived AutoTiling and TIR selection. " +
                    "Residual buffer aliases must be materialized as TIR.Buffer descriptors before PyNTT codegen.");
            }

            var traceLabel = GetPrimFunctionCallTraceLabel(callee.Name);
            var tensorSourceSplitAxes = BuildDeviceFunctionActualTensorSourceSplitAxes(callee, args);
            var (definition, buildResult, wasAdded) = GetOrBuildDeviceFunctionDefinition(callee, tensorSourceSplitAxes);
            _nestedBlockLocalDataPoolBytes = Math.Max(_nestedBlockLocalDataPoolBytes, buildResult.BlockLocalDataPoolBytes);
            _nestedSharedMemoryBytes = Math.Max(_nestedSharedMemoryBytes, buildResult.SharedMemoryBytes);
            if (wasAdded)
            {
                _deviceFunctions.AddRange(buildResult.NestedDeviceFunctions);
                _deviceFunctions.Add(buildResult.Function);
                _helperCalls.AddRange(buildResult.HelperCalls);
                foreach (var opKind in buildResult.OpKinds)
                {
                    _opKinds.Add(opKind);
                }

                if (buildResult.RequiresGridBarrier)
                {
                    _attrs["requires_grid_barrier"] = true;
                }
            }

            var callArguments = BuildDeviceFunctionInvocationArguments(callee, args, definition);
            WriteTraceMarker($"begin_function:{traceLabel}");
            WriteControlLine(BuildDeviceFunctionCallPlaceholder(definition.Name, callArguments));
            WriteTraceMarker($"end_function:{traceLabel}");
            TrackPrimFunctionCallTensorOutputs(callee, args);
            TrackPrimFunctionCallObjectAliases(callee, args, definition);
        }

        private (DeviceFunctionDefinition Definition, DeviceFunctionBuildResult BuildResult, bool WasAdded) GetOrBuildDeviceFunctionDefinition(
            PrimFunction callee,
            IReadOnlyDictionary<IVar, int[][]> tensorSourceSplitAxes)
        {
            if (_deviceFunctionDefinitions.TryGetValue(callee, out var existing))
            {
                ValidateCompatibleTensorSourceSplitAxes(callee, tensorSourceSplitAxes, existing);
                return (existing, existing.BuildResult, false);
            }

            var deviceFunctionName = _sharedHelperRegistry.GetDeviceFunctionName(_function.Name, callee.Name);

            if (_deviceFunctionDefinitionsByName.TryGetValue(deviceFunctionName, out var existingByName))
            {
                ValidateCompatibleDeviceFunctionDefinition(callee, existingByName);
                ValidateCompatibleTensorSourceSplitAxes(callee, tensorSourceSplitAxes, existingByName);
                _deviceFunctionDefinitions.Add(callee, existingByName);
                return (existingByName, existingByName.BuildResult, false);
            }

            var addedActiveFunction = _activePrimFunctionCalls.Add(callee);
            var addedActiveName = _activeDeviceFunctionNames.Add(deviceFunctionName);
            if (!addedActiveFunction || !addedActiveName)
            {
                if (addedActiveFunction)
                {
                    _activePrimFunctionCalls.Remove(callee);
                }

                if (addedActiveName)
                {
                    _activeDeviceFunctionNames.Remove(deviceFunctionName);
                }

                throw new NotSupportedException($"PyNTT PrimFunction call graph contains a recursive call involving {callee.Name}.");
            }

            try
            {
                var formalPlan = BuildDeviceFunctionFormalPlan(callee, deviceFunctionName, tensorSourceSplitAxes);
                var calleeOutputs = GetOutputInfos(callee);
                var deviceAbiState = new KernelAbiState(
                    _inputNames,
                    _kvCacheFieldInputs,
                    _runtimeScalarNames,
                    _abiViewStrideArgNames,
                    _bufferInputIndices,
                    _abiBufferMemo,
                    new HashSet<int>(),
                    new DistributedType?[calleeOutputs.Length],
                    new Dictionary<int, int>(),
                    _activePrimFunctionCalls,
                    _activeDeviceFunctionNames,
                    _deviceFunctionDefinitions,
                    _deviceFunctionDefinitionsByName);
                var deviceFunction = new PyNTTPrimFunctionSourceVisitor(
                    _function,
                    callee.Body,
                    formalPlan.ParameterNames,
                    calleeOutputs,
                    _targetOptions,
                    _sharedHelperRegistry,
                    deviceAbiState,
                    deviceFunctionName,
                    validateOutputs: false,
                    currentFunction: callee,
                    formalTensorParameterBaseNames: formalPlan.TensorBaseNames,
                    formalTensorParameterPoolStrideNames: formalPlan.TensorPoolStrideNames,
                    formalTensorParameterPoolScopeSizeNames: formalPlan.TensorPoolScopeSizeNames,
                    formalTensorParameterDimensions: formalPlan.TensorDimensions,
                    formalTensorParameterGlobalOffsets: formalPlan.TensorGlobalOffsets,
                    formalTensorParameterSourceSplitAxes: formalPlan.TensorSourceSplitAxes,
                    formalDimParameterNames: formalPlan.DimParameterNames,
                    formalObjectParameterBaseNames: formalPlan.ObjectBaseNames,
                    formalWorkspaceParameters: formalPlan.WorkspaceParameters,
                    extraWorkspaceBaseNames: formalPlan.ExtraParameters,
                    extraPointerParameterTritonTypes: formalPlan.ExtraPointerParameterTritonTypes,
                    dataBaseName: formalPlan.DataBaseName,
                    chipLocalDataBaseName: formalPlan.ChipLocalDataBaseName,
                    blockLocalDataBaseName: formalPlan.BlockLocalDataBaseName)
                    .BuildDeviceFunction(deviceFunctionName, new Dictionary<string, string>(StringComparer.Ordinal), new Dictionary<string, string>(StringComparer.Ordinal));

                var definition = new DeviceFunctionDefinition(deviceFunctionName, deviceFunction, formalPlan.Parameters, formalPlan.TensorSourceSplitAxes);
                _deviceFunctionDefinitions.Add(callee, definition);
                _deviceFunctionDefinitionsByName.Add(deviceFunctionName, definition);
                return (definition, deviceFunction, true);
            }
            finally
            {
                if (addedActiveFunction)
                {
                    _activePrimFunctionCalls.Remove(callee);
                }

                if (addedActiveName)
                {
                    _activeDeviceFunctionNames.Remove(deviceFunctionName);
                }
            }
        }

        private static void ValidateCompatibleDeviceFunctionDefinition(PrimFunction callee, DeviceFunctionDefinition existing)
        {
            var parameters = callee.Parameters.ToArray();
            if (parameters.Length != existing.Parameters.Count)
            {
                throw new NotSupportedException($"PyNTT device function name collision for {existing.Name}: existing ABI has {existing.Parameters.Count} parameters, but PrimFunction {callee.Name} has {parameters.Length}.");
            }

            for (var i = 0; i < parameters.Length; i++)
            {
                var parameter = parameters[i];
                var existingParameter = existing.Parameters[i];
                var kind = GetDeviceFunctionFormalParameterKind(parameter);
                if (kind != existingParameter.Kind)
                {
                    throw new NotSupportedException($"PyNTT device function name collision for {existing.Name}: parameter {i} has kind {kind}, but existing ABI has {existingParameter.Kind}.");
                }

                if (!string.Equals(parameter.Name, existingParameter.Parameter.Name, StringComparison.Ordinal))
                {
                    throw new NotSupportedException($"PyNTT device function name collision for {existing.Name}: parameter {i} is named {parameter.Name}, but existing ABI uses {existingParameter.Parameter.Name}.");
                }

                var parameterType = CompilerServices.Print(parameter.CheckedType);
                var existingType = CompilerServices.Print(existingParameter.Parameter.CheckedType);
                if (!string.Equals(parameterType, existingType, StringComparison.Ordinal))
                {
                    throw new NotSupportedException($"PyNTT device function name collision for {existing.Name}: parameter {parameter.Name} has type {parameterType}, but existing ABI uses {existingType}.");
                }

                if (parameter is BufferVar { Role: BufferVarRole.Workspace } workspace &&
                    existingParameter.WorkspaceLocation != workspace.Location)
                {
                    throw new NotSupportedException($"PyNTT device function name collision for {existing.Name}: workspace parameter {parameter.Name} uses {workspace.Location}, but existing ABI uses {existingParameter.WorkspaceLocation}.");
                }

                if (parameter is BufferVar bufferVar &&
                    existingParameter.Parameter is BufferVar existingBufferVar &&
                    bufferVar.LayoutAnnotation != existingBufferVar.LayoutAnnotation)
                {
                    throw new NotSupportedException(
                        $"PyNTT device function name collision for {existing.Name}: parameter {parameter.Name} uses layout {bufferVar.LayoutAnnotation}, " +
                        $"but existing ABI uses {existingBufferVar.LayoutAnnotation}.");
                }
            }
        }

        private static void ValidateCompatibleTensorSourceSplitAxes(
            PrimFunction callee,
            IReadOnlyDictionary<IVar, int[][]> requested,
            DeviceFunctionDefinition existing)
        {
            var parameters = callee.Parameters.ToArray();
            var matchedParameterCount = 0;
            for (var index = 0; index < parameters.Length; index++)
            {
                var parameter = parameters[index];
                if (!requested.TryGetValue(parameter, out var requestedSplitAxes))
                {
                    continue;
                }

                matchedParameterCount++;
                var existingParameter = existing.Parameters[index].Parameter;
                if (!existing.TensorSourceSplitAxes.TryGetValue(existingParameter, out var existingSplitAxes))
                {
                    throw new NotSupportedException($"PyNTT device function {existing.Name} has no tensor source split metadata for ABI parameter {index} ({parameter.Name}) in PrimFunction {callee.Name}.");
                }

                if (!AreSplitAxesEqual(requestedSplitAxes, existingSplitAxes))
                {
                    throw new NotSupportedException($"PyNTT device function {existing.Name} for PrimFunction {callee.Name} is called with incompatible source sharding for ABI parameter {index} ({parameter.Name}): existing {FormatSplitAxes(existingSplitAxes)}, requested {FormatSplitAxes(requestedSplitAxes)}.");
                }
            }

            if (matchedParameterCount != requested.Count)
            {
                throw new InvalidOperationException(
                    $"PyNTT PrimFunction {callee.Name} source split metadata contains {requested.Count - matchedParameterCount} parameter(s) outside its ABI.");
            }
        }

        private IReadOnlyDictionary<IVar, int[][]> BuildDeviceFunctionActualTensorSourceSplitAxes(PrimFunction callee, IReadOnlyList<BaseExpr> args)
        {
            var layouts = new Dictionary<IVar, int[][]>(ReferenceEqualityComparer.Instance);
            var parameters = callee.Parameters.ToArray();
            for (var i = 0; i < parameters.Length; i++)
            {
                var parameter = parameters[i];
                if (GetDeviceFunctionFormalParameterKind(parameter) != DeviceFunctionFormalParameterKind.Tensor)
                {
                    continue;
                }

                var argument = NormalizeParameterAlias(parameter, ResolveBoundExpression(args[i]));
                var buffer = GetBufferOperand(argument, $"PyNTT call to {callee.Name} tensor parameter {parameter.Name} source layout");
                layouts.Add(parameter, GetBufferSourceSplitAxes(buffer, buffer.Rank));
            }

            return layouts;
        }

        private static bool AreSplitAxesEqual(IReadOnlyList<int[]> lhs, IReadOnlyList<int[]> rhs)
        {
            if (lhs.Count != rhs.Count)
            {
                return false;
            }

            for (var i = 0; i < lhs.Count; i++)
            {
                if (!lhs[i].SequenceEqual(rhs[i]))
                {
                    return false;
                }
            }

            return true;
        }

        private static string FormatSplitAxes(IReadOnlyList<int[]> splitAxes)
            => "[" + string.Join(",", splitAxes.Select(axis => "[" + string.Join(",", axis.Select(value => value.ToString(CultureInfo.InvariantCulture))) + "]")) + "]";

        private static DeviceFunctionFormalParameterKind GetDeviceFunctionFormalParameterKind(IVar parameter)
        {
            if (parameter is DimVar)
            {
                return DeviceFunctionFormalParameterKind.Scalar;
            }

            if (parameter is BufferVar { Role: BufferVarRole.Workspace })
            {
                return DeviceFunctionFormalParameterKind.Workspace;
            }

            if (IsObjectDataType(parameter.CheckedDataType))
            {
                return DeviceFunctionFormalParameterKind.Object;
            }

            if (parameter is BufferVar)
            {
                return DeviceFunctionFormalParameterKind.Tensor;
            }

            throw new NotSupportedException($"PyNTT device function parameter {parameter.Name} has unsupported kind {parameter.GetType().Name} and type {parameter.CheckedType}.");
        }

        private DeviceFunctionFormalPlan BuildDeviceFunctionFormalPlan(
            PrimFunction callee,
            string deviceFunctionName,
            IReadOnlyDictionary<IVar, int[][]> tensorSourceSplitAxes)
        {
            var parameterNames = new Dictionary<IVar, string>(ReferenceEqualityComparer.Instance);
            var tensorBaseNames = new Dictionary<IVar, string>(ReferenceEqualityComparer.Instance);
            var tensorPoolStrideNames = new Dictionary<IVar, string>(ReferenceEqualityComparer.Instance);
            var tensorPoolScopeSizeNames = new Dictionary<IVar, string>(ReferenceEqualityComparer.Instance);
            var tensorDimensions = new Dictionary<IVar, PyNTTDimExpression[]>(ReferenceEqualityComparer.Instance);
            var tensorGlobalOffsets = new Dictionary<IVar, PyNTTDimExpression[]>(ReferenceEqualityComparer.Instance);
            var formalTensorSourceSplitAxes = new Dictionary<IVar, int[][]>(ReferenceEqualityComparer.Instance);
            var objectBaseNames = new Dictionary<IVar, string>(ReferenceEqualityComparer.Instance);
            var dimParameterNames = new Dictionary<string, string>(StringComparer.Ordinal);
            var workspaceParameters = new HashSet<IVar>(ReferenceEqualityComparer.Instance);
            var extraParameters = new SortedSet<string>(StringComparer.Ordinal);
            var extraPointerParameterTritonTypes = new Dictionary<string, string>(StringComparer.Ordinal);
            var parameters = new List<DeviceFunctionFormalParameter>();
            var dataBaseName = $"{deviceFunctionName}_data";
            var chipLocalDataBaseName = $"{deviceFunctionName}_chip_local_data";
            var blockLocalDataBaseName = $"{deviceFunctionName}_block_local_data";
            extraParameters.Add(dataBaseName);
            extraParameters.Add(chipLocalDataBaseName);
            extraParameters.Add(blockLocalDataBaseName);
            extraPointerParameterTritonTypes.Add(dataBaseName, "tl.uint8");
            extraPointerParameterTritonTypes.Add(chipLocalDataBaseName, "tl.uint8");
            extraPointerParameterTritonTypes.Add(blockLocalDataBaseName, "tl.uint8");

            var calleeParameters = callee.Parameters.ToArray();
            for (var i = 0; i < calleeParameters.Length; i++)
            {
                var parameter = calleeParameters[i];
                var baseName = SanitizePythonIdentifier($"{deviceFunctionName}_arg{i.ToString(CultureInfo.InvariantCulture)}_{parameter.Name}");
                parameterNames.Add(parameter, baseName);

                if (parameter is DimVar dimVar)
                {
                    var scalarName = baseName;
                    dimParameterNames[SanitizePythonIdentifier(dimVar.Name)] = scalarName;
                    extraParameters.Add(scalarName);
                    parameters.Add(new DeviceFunctionFormalParameter(i, parameter, DeviceFunctionFormalParameterKind.Scalar, null, null, null, Array.Empty<string>(), Array.Empty<string>(), Array.Empty<string>(), scalarName, null));
                    continue;
                }

                if (parameter is BufferVar { Role: BufferVarRole.Workspace } workspace)
                {
                    workspaceParameters.Add(workspace);
                    parameters.Add(new DeviceFunctionFormalParameter(i, parameter, DeviceFunctionFormalParameterKind.Workspace, null, null, null, Array.Empty<string>(), Array.Empty<string>(), Array.Empty<string>(), null, workspace.Location));
                    continue;
                }

                if (IsObjectDataType(parameter.CheckedDataType))
                {
                    objectBaseNames.Add(parameter, baseName);
                    parameters.Add(new DeviceFunctionFormalParameter(i, parameter, DeviceFunctionFormalParameterKind.Object, null, null, null, Array.Empty<string>(), Array.Empty<string>(), Array.Empty<string>(), baseName, null));
                    continue;
                }

                if (parameter is BufferVar bufferVar)
                {
                    var tensorType = GetTensorType(bufferVar.CheckedType, $"PyNTT device function {callee.Name} parameter {bufferVar.Name}");
                    if (bufferVar.LayoutAnnotation.Kind == BufferLayoutKind.Opaque)
                    {
                        throw new NotSupportedException(
                            $"PyNTT device function {callee.Name} tensor parameter {bufferVar.Name} cannot use an opaque buffer layout.");
                    }

                    var isRuntimeStrided = bufferVar.LayoutAnnotation.Kind == BufferLayoutKind.RuntimeStrided;
                    var poolStrideName = $"{baseName}{PoolStrideElementsSuffix}";
                    var poolScopeSizeName = $"{baseName}{PoolScopeSizeSuffix}";
                    var stridePrefix = GetVectorLaneElementCount(tensorType.DType) == 1
                        ? $"{baseName}_scalar_stride"
                        : $"{baseName}_stride";
                    var strideNames = isRuntimeStrided
                        ? Enumerable.Range(0, tensorType.Shape.Rank)
                            .Select(axis => $"{stridePrefix}{axis.ToString(CultureInfo.InvariantCulture)}")
                            .ToArray()
                        : Array.Empty<string>();
                    var dimensionNames = isRuntimeStrided
                        ? Enumerable.Range(0, tensorType.Shape.Rank)
                            .Select(axis => $"{baseName}_dim{axis.ToString(CultureInfo.InvariantCulture)}")
                            .ToArray()
                        : Array.Empty<string>();
                    var globalOffsetNames = isRuntimeStrided
                        ? Enumerable.Range(0, tensorType.Shape.Rank)
                            .Select(axis => $"{baseName}_global_offset{axis.ToString(CultureInfo.InvariantCulture)}")
                            .ToArray()
                        : Array.Empty<string>();
                    tensorBaseNames.Add(bufferVar, baseName);
                    tensorPoolStrideNames.Add(bufferVar, poolStrideName);
                    tensorPoolScopeSizeNames.Add(bufferVar, poolScopeSizeName);
                    if (isRuntimeStrided)
                    {
                        var formalDimensions = BuildFormalTensorDimensions(
                            dimensionNames,
                            GetFormalTensorLocalShape(bufferVar.CheckedType, $"PyNTT device function {callee.Name} parameter {bufferVar.Name} local shape"),
                            $"PyNTT device function {callee.Name} parameter {bufferVar.Name} dimensions");
                        var formalGlobalOffsets = globalOffsetNames
                            .Select(name => new PyNTTDimExpression(
                                name,
                                name,
                                RangeMin: 0,
                                RangeMax: null)
                            {
                                Equivalence = PyNTTDimEquivalence.FromAtom(name),
                            })
                            .ToArray();
                        tensorDimensions.Add(bufferVar, formalDimensions);
                        tensorGlobalOffsets.Add(bufferVar, formalGlobalOffsets);
                    }

                    formalTensorSourceSplitAxes.Add(
                        bufferVar,
                        tensorSourceSplitAxes.TryGetValue(bufferVar, out var sourceSplitAxes)
                            ? sourceSplitAxes
                            : CreateEmptySplitAxes(tensorType.Shape.Rank));
                    extraParameters.Add(baseName);
                    extraPointerParameterTritonTypes.Add(baseName, GetScalarTritonDType(tensorType.DType));
                    extraParameters.Add(poolStrideName);
                    extraParameters.Add(poolScopeSizeName);
                    foreach (var strideName in strideNames)
                    {
                        extraParameters.Add(strideName);
                    }

                    foreach (var dimensionName in dimensionNames)
                    {
                        extraParameters.Add(dimensionName);
                    }

                    foreach (var globalOffsetName in globalOffsetNames)
                    {
                        extraParameters.Add(globalOffsetName);
                    }

                    parameters.Add(new DeviceFunctionFormalParameter(i, parameter, DeviceFunctionFormalParameterKind.Tensor, baseName, poolStrideName, poolScopeSizeName, strideNames, dimensionNames, globalOffsetNames, null, null));
                    continue;
                }

                throw new NotSupportedException($"PyNTT device function {callee.Name} parameter {parameter.Name} has unsupported kind {parameter.GetType().Name} and type {parameter.CheckedType}.");
            }

            return new(
                parameterNames,
                tensorBaseNames,
                tensorPoolStrideNames,
                tensorPoolScopeSizeNames,
                tensorDimensions,
                tensorGlobalOffsets,
                formalTensorSourceSplitAxes,
                dimParameterNames,
                objectBaseNames,
                workspaceParameters,
                extraParameters,
                extraPointerParameterTritonTypes,
                dataBaseName,
                chipLocalDataBaseName,
                blockLocalDataBaseName,
                parameters);
        }

        private static Shape GetFormalTensorLocalShape(IRType type, string context)
            => type switch
            {
                TensorType tensorType => tensorType.Shape,
                DistributedType distributedType => DistributedUtility.GetDividedTensorType(distributedType).Shape,
                _ => throw new NotSupportedException($"PyNTT requires tensor type for {context}, got {type}."),
            };

        private static PyNTTDimExpression[] BuildFormalTensorDimensions(
            IReadOnlyList<string> dimensionNames,
            Shape localShape,
            string context)
        {
            var localDimensions = GetRankedShapeDimensions(localShape, context);
            if (localDimensions.Length != dimensionNames.Count)
            {
                throw new NotSupportedException($"{context} rank mismatch, local shape rank={localDimensions.Length}, ABI dims={dimensionNames.Count}.");
            }

            var emitter = new PyNTTDimExpressionEmitter();
            return localDimensions
                .Select((dimension, axis) =>
                {
                    var localDimension = emitter.Emit(dimension);
                    return new PyNTTDimExpression(
                        dimensionNames[axis],
                        dimensionNames[axis],
                        RangeMin: localDimension.MinValue,
                        RangeMax: localDimension.MaxValue)
                    {
                        Equivalence = PyNTTDimEquivalence.FromAtom(dimensionNames[axis]),
                    };
                })
                .ToArray();
        }

        private void TrackPrimFunctionCallObjectAliases(PrimFunction callee, IReadOnlyList<BaseExpr> args, DeviceFunctionDefinition definition)
        {
            if (definition.BuildResult.FormalObjectOutputAliases.Count == 0)
            {
                return;
            }

            var parameters = callee.Parameters.ToArray();
            var outputs = PyNTTFunctionOutputs.GetOutputParameters(callee);
            foreach (var pair in definition.BuildResult.FormalObjectOutputAliases)
            {
                if ((uint)pair.Key >= (uint)outputs.Length)
                {
                    throw new NotSupportedException($"PyNTT device function {definition.Name} reports object output alias index {pair.Key}, but {callee.Name} only has {outputs.Length} outputs.");
                }

                var outputParameter = outputs[pair.Key];
                var outputParameterIndex = Array.FindIndex(parameters, parameter => ReferenceEquals(parameter, outputParameter));
                if (outputParameterIndex < 0)
                {
                    throw new NotSupportedException($"PyNTT cannot find output parameter {outputParameter.Name} in PrimFunction {callee.Name}.");
                }

                var sourceParameter = definition.Parameters.SingleOrDefault(parameter =>
                    parameter.Kind == DeviceFunctionFormalParameterKind.Object &&
                    string.Equals(parameter.ObjectBaseName, pair.Value, StringComparison.Ordinal));
                if (sourceParameter is null)
                {
                    throw new NotSupportedException($"PyNTT device function {definition.Name} reports object output {outputParameter.Name} aliases unknown formal object {pair.Value}.");
                }

                VisitObjectAssignment(
                    args[outputParameterIndex],
                    args[sourceParameter.Index],
                    $"PrimFunction {callee.Name} object output {outputParameter.Name}");
            }
        }

        private void TrackPrimFunctionCallTensorOutputs(PrimFunction callee, IReadOnlyList<BaseExpr> args)
        {
            var parameters = callee.Parameters.ToArray();
            foreach (var outputParameter in PyNTTFunctionOutputs.GetOutputParameters(callee))
            {
                if (IsObjectDataType(outputParameter.CheckedDataType))
                {
                    continue;
                }

                var outputParameterIndex = Array.FindIndex(parameters, parameter => ReferenceEquals(parameter, outputParameter));
                if (outputParameterIndex < 0)
                {
                    throw new NotSupportedException($"PyNTT cannot find tensor output parameter {outputParameter.Name} in PrimFunction {callee.Name}.");
                }

                var argument = ResolveBoundExpression(args[outputParameterIndex]);
                var buffer = GetBufferOperand(argument, $"PyNTT PrimFunction {callee.Name} tensor output {outputParameter.Name}");
                MarkStoredOutput(buffer, $"PyNTT PrimFunction {callee.Name} tensor output {outputParameter.Name}");
            }
        }

        private string[] BuildDeviceFunctionInvocationArguments(PrimFunction callee, IReadOnlyList<BaseExpr> args, DeviceFunctionDefinition definition)
        {
            var values = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                [$"{definition.Name}_data"] = _dataBaseName,
                [$"{definition.Name}_chip_local_data"] = _chipLocalDataBaseName,
                [$"{definition.Name}_block_local_data"] = _blockLocalDataBaseName,
            };
            foreach (var parameter in definition.Parameters)
            {
                var argument = NormalizeParameterAlias(parameter.Parameter, ResolveBoundExpression(args[parameter.Index]));
                switch (parameter.Kind)
                {
                    case DeviceFunctionFormalParameterKind.Workspace:
                        if (parameter.WorkspaceLocation is not { } workspaceLocation)
                        {
                            throw new NotSupportedException($"PyNTT device function {definition.Name} workspace parameter {parameter.Parameter.Name} is missing its memory location.");
                        }

                        values[workspaceLocation switch
                        {
                            MemoryLocation.Data => $"{definition.Name}_data",
                            MemoryLocation.ChipLocalData => $"{definition.Name}_chip_local_data",
                            MemoryLocation.BlockLocalData => $"{definition.Name}_block_local_data",
                            var location => throw new NotSupportedException($"PyNTT call to {callee.Name} workspace parameter {parameter.Parameter.Name} cannot use memory location {location}."),
                        }] = BuildWorkspaceBasePointerExpression(callee, (BufferVar)parameter.Parameter, argument);
                        break;
                    case DeviceFunctionFormalParameterKind.Tensor:
                        var buffer = GetBufferOperand(argument, $"PyNTT call to {callee.Name} tensor parameter {parameter.Parameter.Name}");
                        values[RequireParameterName(parameter.BaseName, definition.Name, parameter.Parameter.Name)] = BuildFormalTensorBasePointerArgument(buffer);
                        values[RequireParameterName(parameter.PoolStrideName, definition.Name, parameter.Parameter.Name)] = BuildFormalTensorPoolStrideElementsArgument(buffer);
                        values[RequireParameterName(parameter.PoolScopeSizeName, definition.Name, parameter.Parameter.Name)] = BuildFormalTensorPoolScopeSizeArgument(buffer);
                        if (parameter.StrideNames.Length != 0)
                        {
                            var strides = GetBufferStrides(buffer);
                            if (strides.Length != parameter.StrideNames.Length)
                            {
                                throw new NotSupportedException($"PyNTT call to {callee.Name} tensor parameter {parameter.Parameter.Name} expects {parameter.StrideNames.Length} strides, got {strides.Length}.");
                            }

                            for (var i = 0; i < strides.Length; i++)
                            {
                                values[parameter.StrideNames[i]] = strides[i].TritonExpression;
                            }
                        }

                        if (parameter.DimensionNames.Length != 0)
                        {
                            var activeShape = GetBufferActiveShape(buffer);
                            if (activeShape.Length != parameter.DimensionNames.Length)
                            {
                                throw new NotSupportedException($"PyNTT call to {callee.Name} tensor parameter {parameter.Parameter.Name} expects {parameter.DimensionNames.Length} dimensions, got {activeShape.Length}.");
                            }

                            for (var i = 0; i < activeShape.Length; i++)
                            {
                                values[parameter.DimensionNames[i]] = activeShape[i].TritonExpression;
                            }
                        }

                        if (parameter.GlobalOffsetNames.Length != 0)
                        {
                            var globalOffsets = GetBufferGlobalOffsets(buffer);
                            if (globalOffsets.Length != parameter.GlobalOffsetNames.Length)
                            {
                                throw new NotSupportedException($"PyNTT call to {callee.Name} tensor parameter {parameter.Parameter.Name} expects {parameter.GlobalOffsetNames.Length} global offsets, got {globalOffsets.Length}.");
                            }

                            for (var i = 0; i < globalOffsets.Length; i++)
                            {
                                values[parameter.GlobalOffsetNames[i]] = globalOffsets[i].TritonExpression;
                            }
                        }

                        break;
                    case DeviceFunctionFormalParameterKind.Scalar:
                        values[RequireParameterName(parameter.ScalarName, definition.Name, parameter.Parameter.Name)] = BuildScalarExpression(argument);
                        break;
                    case DeviceFunctionFormalParameterKind.Object:
                        var objectBaseName = RequireParameterName(parameter.ObjectBaseName, definition.Name, parameter.Parameter.Name);
                        var prefix = objectBaseName + "_";
                        foreach (var pair in definition.BuildResult.FormalObjectFieldStorages)
                        {
                            if (!pair.Key.StartsWith(prefix, StringComparison.Ordinal))
                            {
                                continue;
                            }

                            var field = pair.Key[prefix.Length..];
                            values[pair.Key] = GetKVCacheFieldArgument(argument, field, pair.Value);
                        }

                        break;
                    default:
                        throw new NotSupportedException($"PyNTT call to {callee.Name} has unsupported formal parameter kind {parameter.Kind}.");
                }
            }

            return definition.BuildResult.Function.ExtraParameters
                .Select(parameter =>
                {
                    if (!values.TryGetValue(parameter, out var value))
                    {
                        throw new NotSupportedException($"PyNTT call to {callee.Name} did not bind device function parameter {parameter}.");
                    }

                    return value;
                })
                .ToArray();
        }

        private static string RequireParameterName(string? name, string deviceFunctionName, string sourceParameterName)
            => string.IsNullOrWhiteSpace(name)
                ? throw new NotSupportedException($"PyNTT device function {deviceFunctionName} has incomplete formal binding for parameter {sourceParameterName}.")
                : name;

        private string BuildWorkspaceBasePointerExpression(PrimFunction callee, BufferVar workspace, BaseExpr argument)
        {
            if (argument is not TIR.Buffer buffer)
            {
                throw new NotSupportedException($"PyNTT call to {callee.Name} workspace parameter {workspace.Name} expects a caller workspace buffer, got {argument.GetType().Name}.");
            }

            if (buffer.ElemType != DataTypes.UInt8)
            {
                throw new NotSupportedException($"PyNTT call to {callee.Name} workspace parameter {workspace.Name} expects a u8 workspace buffer, got {buffer.ElemType}.");
            }

            if (buffer.MemSpan.Buffer.Location != workspace.Location)
            {
                throw new NotSupportedException($"PyNTT call to {callee.Name} workspace parameter {workspace.Name} expects {workspace.Location} workspace, got {buffer.MemSpan.Buffer.Location}.");
            }

            var offsetBytes = GetBufferOffsetBytes(buffer);
            if (RequiresShardCoords(offsetBytes))
            {
                throw new NotSupportedException($"PyNTT workspace buffer {buffer.Name} has a shard-dependent base offset, which cannot be passed as a callee workspace base pointer.");
            }

            var baseName = buffer.MemSpan.Buffer.Location switch
            {
                MemoryLocation.Data => GetDataBaseName(buffer),
                MemoryLocation.ChipLocalData => GetChipLocalDataBaseName(buffer),
                MemoryLocation.BlockLocalData => GetBlockLocalDataBaseName(buffer),
                var location => throw new NotSupportedException($"PyNTT workspace buffer {buffer.Name} cannot use memory location {location}."),
            };
            return BuildPointerExpression(new BufferRef(baseName, offsetBytes, "0", null, null, true), "tl.uint8");
        }

        private static BaseExpr NormalizeParameterAlias(IVar parameter, BaseExpr alias)
        {
            alias = UnwrapInputBoxing(alias);
            if (parameter is DimVar)
            {
                return alias switch
                {
                    Dimension dimension => dimension,
                    TensorConst tensorConst when tensorConst.Value.Shape.IsScalar => new DimConst(ReadScalarInt64(tensorConst.Value, $"PyNTT dimension parameter {parameter.Name}")),
                    _ => throw new NotSupportedException($"PyNTT cannot bind dimension parameter {parameter.Name} to call argument {alias.GetType().Name}."),
                };
            }

            return alias;
        }

        private void VisitTensorLoad(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count != 2)
            {
                throw new NotSupportedException("PyNTT TensorLoad codegen expects (dest_buffer, source_tensor).");
            }

            var destExpr = UnwrapInputBoxing(args[0]);
            var srcExpr = UnwrapInputBoxing(args[1]);
            if (IsObjectExpression(destExpr) || IsObjectExpression(srcExpr))
            {
                VisitObjectAssignment(destExpr, srcExpr, "TensorLoad");
                return;
            }

            var dest = GetBufferOperand(destExpr, "PyNTT TensorLoad destination");

            if (srcExpr is TIR.Buffer srcBuffer)
            {
                VisitInternalTensorLoad(dest, srcBuffer);
                return;
            }

            if (TryGetFormalTensorBuffer(srcExpr, "PyNTT TensorLoad formal source", out var formalSrc))
            {
                VisitInternalTensorLoad(dest, formalSrc);
                return;
            }

            var inputIndex = GetInputIndex(srcExpr);
            _bufferInputIndices[dest] = inputIndex;
            if (IsObjectDataType(dest.ElemType))
            {
                return;
            }

            var localShape = GetBufferShape(dest);
            var globalShape = GetTensorShape(srcExpr, $"TensorLoad source input{inputIndex}");
            var helperName = GetNextHelperName("tensor_load");
            WriteHelperTemplate(
                "triton/kernels/TensorLoad.py.jinja",
                new PyNTTTensorLoadTemplateModel(
                    helperName,
                    $"input{inputIndex}",
                    0,
                    GetBufferPointer(dest),
                    GetPyNTTDTypeName(dest.ElemType),
                    GetTritonDType(dest.ElemType),
                    localShape,
                    GetBufferStrides(dest),
                    globalShape,
                    GetBufferGlobalOffsets(dest),
                    GetHierarchy(dest),
                    GetBufferSplitAxes(dest, globalShape.Length),
                    GetVectorLaneElementCount(dest.ElemType),
                    GetVectorLanes(dest.ElemType),
                    $"TensorLoad -> {dest.Name}"));
            WriteHelperInvocation(helperName, $"input{inputIndex}", $"input{inputIndex}_pool_stride_elements");
        }

        private void VisitTensorStore(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count != 2)
            {
                throw new NotSupportedException("PyNTT TensorStore codegen expects (source_buffer, dest_tensor).");
            }

            var src = GetBufferOperand(args[0], "PyNTT TensorStore source");

            if (args[1] is TIR.Buffer destBuffer)
            {
                if (destBuffer.MemSpan.Buffer.Location == MemoryLocation.Output && !IsFormalTensorBuffer(destBuffer))
                {
                    throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} must store to output BufferVar ABI parameters, but TensorStore destination {destBuffer.Name} is a TIR Buffer in Output memory.");
                }

                VisitInternalTensorStore(src, destBuffer);
                return;
            }

            if (TryGetFormalTensorBuffer(args[1], "PyNTT TensorStore formal destination", out var formalDest))
            {
                VisitInternalTensorStore(src, formalDest);
                return;
            }

            _ = GetBufferShape(src);
            var globalShape = GetTensorShape(args[1], "TensorStore destination");
            var outputIndex = GetOutputIndex(args[1]);
            WriteTensorStore(src, outputIndex, globalShape, $"{src.Name} -> TensorStore");
        }

        private void VisitInternalTensorLoad(TIR.Buffer dest, TIR.Buffer src)
        {
            VisitInternalTensorCopy(src, dest, "TensorLoad");
        }

        private void VisitInternalTensorStore(TIR.Buffer src, TIR.Buffer dest)
        {
            VisitInternalTensorCopy(src, dest, "TensorStore");
        }

        private void VisitTileLoad(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count != 2)
            {
                throw new NotSupportedException("PyNTT TileLoad codegen expects (destination, source).");
            }

            var dest = GetBufferOperand(args[0], "PyNTT TileLoad destination");
            var src = GetBufferOperand(args[1], "PyNTT TileLoad source");
            VisitInternalTensorCopy(src, dest, "TileLoad");
        }

        private void VisitTileStore(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count != 2)
            {
                throw new NotSupportedException("PyNTT TileStore codegen expects (source, destination).");
            }

            var src = GetBufferOperand(args[0], "PyNTT TileStore source");
            var dest = GetBufferOperand(args[1], "PyNTT TileStore destination");
            VisitInternalTensorCopy(src, dest, "TileStore");
        }

        private void VisitInternalTensorCopy(TIR.Buffer src, TIR.Buffer dest, string operation)
        {
            if (IsObjectDataType(src.ElemType) || IsObjectDataType(dest.ElemType))
            {
                return;
            }

            ValidateMatchingBufferDType($"PyNTT {operation} buffer source/destination", src, dest);
            var sourceShape = GetBufferActiveShape(src);
            var destinationShape = GetBufferActiveShape(dest);
            var sourceGlobalOffsets = GetBufferGlobalOffsets(src);
            var destinationGlobalOffsets = GetBufferGlobalOffsets(dest);
            ValidateSameRank($"PyNTT {operation} source/destination", sourceShape, destinationShape);
            ValidateSameRank($"PyNTT {operation} source offsets", sourceShape, sourceGlobalOffsets);
            ValidateSameRank($"PyNTT {operation} destination offsets", destinationShape, destinationGlobalOffsets);
            ValidateSameShape(
                $"PyNTT {operation} global shape",
                GetBufferGlobalShape(src),
                GetBufferGlobalShape(dest));
            var sourceStrides = GetBufferStrides(src);
            var destinationStrides = GetBufferStrides(dest);
            var vectorLaneShape = GetVectorLanes(src.ElemType);
            var copyPlan = BuildRegionCopyPlan(
                sourceShape,
                destinationShape,
                sourceGlobalOffsets,
                destinationGlobalOffsets,
                sourceStrides,
                destinationStrides,
                vectorLaneShape);
            var helperName = GetNextHelperName(GetTensorCopyHelperKind(operation, src, dest));
            WriteHelperTemplate(
                "triton/kernels/TensorRegionCopy.py.jinja",
                new PyNTTRegionCopyTemplateModel(
                    helperName,
                    GetRegionCopyBufferPointer(src),
                    GetRegionCopyBufferPointer(dest),
                    GetPyNTTDTypeName(src.ElemType),
                    GetScalarTritonDType(src.ElemType),
                    sourceShape,
                    destinationShape,
                    sourceGlobalOffsets,
                    destinationGlobalOffsets,
                    sourceStrides,
                    destinationStrides,
                    vectorLaneShape,
                    operation,
                    copyPlan,
                    $"{operation}: {src.Name} -> {dest.Name}"));
            WriteLine(BuildHelperCall(helperName));
            MarkStoredOutput(dest, $"PyNTT {operation}");
        }

        private static PyNTTRegionCopyPlanTemplateModel BuildRegionCopyPlan(
            IReadOnlyList<PyNTTDimExpression> sourceShape,
            IReadOnlyList<PyNTTDimExpression> destinationShape,
            IReadOnlyList<PyNTTDimExpression> sourceGlobalOffsets,
            IReadOnlyList<PyNTTDimExpression> destinationGlobalOffsets,
            IReadOnlyList<PyNTTDimExpression> sourceStrides,
            IReadOnlyList<PyNTTDimExpression> destinationStrides,
            IReadOnlyList<int> vectorLaneShape)
        {
            var rank = sourceShape.Count;
            if (destinationShape.Count != rank ||
                sourceGlobalOffsets.Count != rank ||
                destinationGlobalOffsets.Count != rank ||
                sourceStrides.Count != rank ||
                destinationStrides.Count != rank)
            {
                throw new ArgumentException("PyNTT region-copy plan requires equal-rank buffer metadata.");
            }

            var copyExtents = new PyNTTDimExpression[rank];
            var sourceOrigins = new PyNTTDimExpression[rank];
            var destinationOrigins = new PyNTTDimExpression[rank];
            for (var axis = 0; axis < rank; axis++)
            {
                var copyStart = MaxDims(sourceGlobalOffsets[axis], destinationGlobalOffsets[axis]);
                var sourceEnd = AddDims(sourceGlobalOffsets[axis], sourceShape[axis]);
                var destinationEnd = AddDims(destinationGlobalOffsets[axis], destinationShape[axis]);
                var copyEnd = MinDims(sourceEnd, destinationEnd);

                var extent = SubtractDims(copyEnd, copyStart);
                if (EquivalentDim(extent, sourceShape[axis]))
                {
                    extent = sourceShape[axis];
                }
                else if (EquivalentDim(extent, destinationShape[axis]))
                {
                    extent = destinationShape[axis];
                }
                else if (extent.MinValue is null || extent.MinValue < 0)
                {
                    extent = MaxDims(extent, PyNTTDimExpression.Zero);
                }

                var sourceOrigin = SubtractDims(copyStart, sourceGlobalOffsets[axis]);
                if (EquivalentDim(sourceOrigin, PyNTTDimExpression.Zero))
                {
                    sourceOrigin = PyNTTDimExpression.Zero;
                }

                var destinationOrigin = SubtractDims(copyStart, destinationGlobalOffsets[axis]);
                if (EquivalentDim(destinationOrigin, PyNTTDimExpression.Zero))
                {
                    destinationOrigin = PyNTTDimExpression.Zero;
                }

                copyExtents[axis] = extent;
                sourceOrigins[axis] = sourceOrigin;
                destinationOrigins[axis] = destinationOrigin;
            }

            var extents = new List<PyNTTDimExpression>(rank + vectorLaneShape.Count);
            extents.AddRange(copyExtents);
            for (var axis = 0; axis < vectorLaneShape.Count; axis++)
            {
                if (vectorLaneShape[axis] <= 0)
                {
                    throw new NotSupportedException(
                        $"PyNTT region-copy vector lane {axis} must be positive, got {vectorLaneShape[axis]}.");
                }

                extents.Add(ToDim(vectorLaneShape[axis]));
            }

            var coversWholeSource = sourceOrigins.All(origin => EquivalentDim(origin, PyNTTDimExpression.Zero)) &&
                copyExtents.Zip(sourceShape).All(pair => EquivalentDim(pair.First, pair.Second));
            var coversWholeDestination = destinationOrigins.All(origin => EquivalentDim(origin, PyNTTDimExpression.Zero)) &&
                copyExtents.Zip(destinationShape).All(pair => EquivalentDim(pair.First, pair.Second));
            return new(
                sourceOrigins,
                destinationOrigins,
                extents.ToArray(),
                coversWholeSource,
                coversWholeDestination);
        }

        private static void ValidateMatchingBufferDType(string context, TIR.Buffer src, TIR.Buffer dest)
        {
            ValidateMatchingVectorLanes(context, src.ElemType, dest.ElemType);
            var srcScalarDType = GetPyNTTScalarDTypeName(src.ElemType);
            var destScalarDType = GetPyNTTScalarDTypeName(dest.ElemType);
            if (srcScalarDType != destScalarDType)
            {
                throw new NotSupportedException($"{context} expects matching scalar dtypes, got source {srcScalarDType} and destination {destScalarDType}.");
            }
        }

        private static string GetTensorCopyHelperKind(string operation, TIR.Buffer src, TIR.Buffer dest)
        {
            var (action, semanticBuffer) = operation switch
            {
                "TensorLoad" => ("tensor_load", dest),
                "TensorStore" => ("tensor_store", src),
                "TileLoad" => ("tile_load", dest),
                "TileStore" => ("tile_store", src),
                _ => throw new InvalidOperationException($"Unsupported PyNTT tensor copy operation {operation}."),
            };
            return $"{action}_{GetDiagnosticBufferName(semanticBuffer)}";
        }

        private static string GetDiagnosticBufferName(TIR.Buffer buffer)
        {
            var name = buffer.Name;
            foreach (var prefix in new[] { "alias_buffer_", "alias_view_", "buffer_", "view_" })
            {
                if (name.StartsWith(prefix, StringComparison.Ordinal))
                {
                    name = name[prefix.Length..];
                    break;
                }
            }

            var ownerSeparator = name.IndexOf("__at_", StringComparison.Ordinal);
            if (ownerSeparator >= 0)
            {
                name = name[..ownerSeparator];
            }

            return string.IsNullOrWhiteSpace(name) ? "buffer" : name;
        }

        private void WriteTensorStore(TIR.Buffer src, int outputIndex, PyNTTDimExpression[] globalShape, string comment)
        {
            RecordTensorOutputStore(outputIndex, GetDistributedType(src), "TensorStore");
            var outputName = _outputs[outputIndex].Name;
            var helperName = GetNextHelperName("output_tensor_store");
            var localShape = GetBufferShape(src);
            WriteHelperTemplate(
                "triton/kernels/TensorStore.py.jinja",
                new PyNTTTensorStoreTemplateModel(
                    helperName,
                    GetBufferPointer(src),
                    outputName,
                    0,
                    GetPyNTTDTypeName(src.ElemType),
                    GetTritonDType(src.ElemType),
                    localShape,
                    GetBufferStrides(src),
                    globalShape,
                    GetBufferGlobalOffsets(src),
                    GetHierarchy(src),
                    GetBufferSplitAxes(src, globalShape.Length),
                    GetVectorLaneElementCount(src.ElemType),
                    GetVectorLanes(src.ElemType),
                    comment));
            WriteLine(BuildHelperCall(helperName, outputName, $"{outputName}_pool_stride_elements"));
        }

        private void VisitMemcopy(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count != 2)
            {
                throw new NotSupportedException("PyNTT Memcopy codegen expects (destination, source).");
            }

            var dest = UnwrapInputBoxing(args[0]);
            var src = UnwrapInputBoxing(args[1]);
            if (IsObjectExpression(dest) || IsObjectExpression(src))
            {
                VisitObjectMemcopy(dest, src);
                return;
            }

            if (dest is IVar && src is IVar)
            {
                VisitTensorOutputAliasMemcopy(dest, src);
                return;
            }

            if (dest is not TIR.Buffer destBuffer || src is not TIR.Buffer srcBuffer)
            {
                throw new NotSupportedException($"PyNTT Memcopy expects object aliases or TIR buffers, got destination {dest.GetType().Name} and source {src.GetType().Name}.");
            }

            ValidateSameShape("PyNTT Memcopy", GetBufferShape(srcBuffer), GetBufferShape(destBuffer));
            ValidateMatchingVectorLanes("PyNTT Memcopy source/destination", srcBuffer.ElemType, destBuffer.ElemType);
            var srcScalarDType = GetPyNTTScalarDTypeName(srcBuffer.ElemType);
            var destScalarDType = GetPyNTTScalarDTypeName(destBuffer.ElemType);
            if (srcScalarDType != destScalarDType)
            {
                throw new NotSupportedException($"PyNTT Memcopy expects matching scalar dtypes, got source {srcScalarDType} and destination {destScalarDType}.");
            }

            SetComputeOp("memcopy");
            _attrs["op"] = "memcopy";
            _attrs["dtype"] = GetPyNTTDTypeName(destBuffer.ElemType);
            _attrs["shape"] = GetBufferShape(destBuffer);
            var helperName = GetNextHelperName("memcopy");
            WriteHelperTemplate(
                "triton/kernels/Memcopy.py.jinja",
                new PyNTTMemcopyTemplateModel(
                    helperName,
                    GetBufferScalarPointer(srcBuffer),
                    GetBufferScalarPointer(destBuffer),
                    destScalarDType,
                    GetScalarTritonDType(destBuffer.ElemType),
                    GetBufferShape(destBuffer),
                    GetBufferStrides(srcBuffer),
                    GetBufferStrides(destBuffer),
                    GetVectorLaneElementCount(destBuffer.ElemType),
                    GetVectorLanes(destBuffer.ElemType),
                    $"{srcBuffer.Name} -> {destBuffer.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitTensorOutputAliasMemcopy(BaseExpr dest, BaseExpr src)
        {
            var outputIndex = GetOutputIndex(dest);
            if (_storedOutputIndices.Contains(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} materializes output {_outputs[outputIndex].Name} both by TensorStore and tensor Memcopy.");
            }

            var inputIndex = GetInputIndex(src);
            var destType = GetTensorType(dest.CheckedType, $"PyNTT tensor Memcopy destination {_outputs[outputIndex].Name}");
            var srcType = GetTensorType(src.CheckedType, $"PyNTT tensor Memcopy source input{inputIndex}");
            if (destType.DType != srcType.DType || destType.Shape.Rank != srcType.Shape.Rank)
            {
                throw new NotSupportedException($"PyNTT tensor Memcopy alias expects matching dtype/rank, got destination {destType} and source {srcType}.");
            }

            if (_outputAliases.TryGetValue(outputIndex, out var existingInputIndex) && existingInputIndex != inputIndex)
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} assigns output {_outputs[outputIndex].Name} to multiple input aliases.");
            }

            _outputAliases[outputIndex] = inputIndex;
        }

        private void VisitObjectAssignment(BaseExpr dest, BaseExpr src, string context)
        {
            if (!IsObjectExpression(dest) || !IsObjectExpression(src))
            {
                throw new NotSupportedException($"PyNTT object {context} expects both operands to be object tensors, got destination {dest.CheckedDataType} and source {src.CheckedDataType}.");
            }

            if (!TryGetOutputIndex(dest, out var outputIndex))
            {
                TrackObjectBufferAlias(dest, src, context);
                return;
            }

            if (TryGetFormalObjectBaseName(src, out var formalObjectBaseName))
            {
                RecordFormalObjectOutputAlias(outputIndex, formalObjectBaseName, context);
                return;
            }

            if (TryResolveObjectInputIndex(src, out var inputIndex))
            {
                RecordRuntimeObjectOutputAlias(outputIndex, inputIndex, context);
                return;
            }

            if (_outputAliases.ContainsKey(outputIndex) || _formalObjectOutputAliases.ContainsKey(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} materializes output {_outputs[outputIndex].Name} both by input alias and object {context}.");
            }

            if (!TryRecordStoredOutput(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} materializes output {_outputs[outputIndex].Name} more than once.");
            }
        }

        private void RecordRuntimeObjectOutputAlias(int outputIndex, int inputIndex, string context)
        {
            if (_storedOutputIndices.Contains(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} materializes output {_outputs[outputIndex].Name} both by TensorStore and object {context}.");
            }

            if (_formalObjectOutputAliases.ContainsKey(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} assigns output {_outputs[outputIndex].Name} to both runtime and formal object aliases.");
            }

            if (_outputAliases.TryGetValue(outputIndex, out var existingInputIndex) && existingInputIndex != inputIndex)
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} assigns output {_outputs[outputIndex].Name} to multiple input aliases.");
            }

            _outputAliases[outputIndex] = inputIndex;
        }

        private void RecordFormalObjectOutputAlias(int outputIndex, string formalObjectBaseName, string context)
        {
            if (_storedOutputIndices.Contains(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} materializes output {_outputs[outputIndex].Name} both by TensorStore and object {context}.");
            }

            if (_outputAliases.ContainsKey(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} assigns output {_outputs[outputIndex].Name} to both runtime and formal object aliases.");
            }

            if (_formalObjectOutputAliases.TryGetValue(outputIndex, out var existingFormalObjectBaseName) &&
                !string.Equals(existingFormalObjectBaseName, formalObjectBaseName, StringComparison.Ordinal))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} assigns output {_outputs[outputIndex].Name} to multiple formal object aliases.");
            }

            _formalObjectOutputAliases[outputIndex] = formalObjectBaseName;
        }

        private void TrackObjectBufferAlias(BaseExpr dest, BaseExpr src, string context)
        {
            dest = UnwrapInputBoxing(dest);
            if (dest is not TIR.Buffer destBuffer || !IsObjectDataType(destBuffer.ElemType))
            {
                return;
            }

            if (TryGetFormalObjectBaseName(src, out var formalObjectBaseName))
            {
                RecordObjectFormalAlias(destBuffer, formalObjectBaseName);
                return;
            }

            if (TryResolveObjectInputIndex(src, out var inputIndex))
            {
                RecordObjectInputAlias(destBuffer, inputIndex);
                return;
            }

            throw new NotSupportedException($"PyNTT object {context} cannot track alias source {src.GetType().Name} for intermediate object buffer {destBuffer.Name}.");
        }

        private bool TryResolveObjectInputIndex(BaseExpr expr, out int inputIndex)
        {
            expr = UnwrapInputBoxing(expr);
            if (TryGetFormalObjectBaseName(expr, out _))
            {
                inputIndex = -1;
                return false;
            }

            if (TryGetDirectInputName(expr, out _))
            {
                inputIndex = GetInputIndex(expr);
                return true;
            }

            if (expr is TIR.Buffer buffer)
            {
                foreach (var aliasBuffer in EnumerateObjectAliasBuffers(buffer))
                {
                    if (_bufferInputIndices.TryGetValue(aliasBuffer, out inputIndex))
                    {
                        return true;
                    }
                }
            }

            inputIndex = -1;
            return false;
        }

        private void VisitObjectMemcopy(BaseExpr dest, BaseExpr src)
        {
            VisitObjectAssignment(dest, src, "Memcopy");
        }

        private void VisitUnary(Nncase.TIR.NTT.Unary unary, IReadOnlyList<BaseExpr> args)
        {
            VisitElementwiseUnary(unary.UnaryOp, "PyNTT Unary", args);
        }

        private void VisitElementwiseUnary(UnaryOp unaryOp, string context, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException($"{context} codegen expects TIR buffer operands.");
            }

            SetComputeOp("unary");
            var shape = GetBufferShape(output);
            var inputShape = GetBufferShape(input);
            ValidateSameShape(context, inputShape, shape);
            ValidateMatchingVectorLanes($"{context} input/output", input.ElemType, output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            _attrs["op"] = GetOpName(unaryOp);
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_unary");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseUnary.py.jinja",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(input.ElemType),
                    GetVectorLanes(output.ElemType),
                    shape,
                    GetUnaryExpression(unaryOp),
                    (string)_attrs["op"],
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void VisitExpand(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Expand codegen expects input and output TIR buffers.");
            }

            SetComputeOp("expand");
            var shape = GetBufferShape(output);
            var inputShape = GetBufferShape(input);
            ValidateBroadcastable("PyNTT Expand input", inputShape, shape);
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT Expand input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Expand output");
            if (inputVectorLaneCount != 1 && inputVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT Expand expects input vector lanes to be scalar or match output lanes, got input={inputVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            _attrs["op"] = "expand";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("expand_compute");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseUnary.py.jinja",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(input.ElemType),
                    GetVectorLanes(output.ElemType),
                    shape,
                    "value0",
                    "expand",
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void VisitGather(Nncase.TIR.NTT.Gather gather, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer index ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Gather codegen expects input, index, and output TIR buffers.");
            }

            SetComputeOp("gather");
            var inputShape = GetBufferShape(input);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var indexShape = GetBufferShape(index);
            var outputShape = GetBufferShape(output);
            var axis = NormalizeAxis(gather.Axis, inputShape.Length, "PyNTT Gather");
            var inputVectorLanes = GetVectorLanes(input.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            var valueVectorLaneCount = 1;
            if (inputVectorLanes.Length != 0 || outputVectorLanes.Length != 0)
            {
                if (inputVectorLanes.Length != 1 || outputVectorLanes.Length != 1 || inputVectorLanes[0] != outputVectorLanes[0])
                {
                    throw new NotSupportedException($"PyNTT Gather currently supports matching one-dimensional input/output vector lanes only, got input lanes [{string.Join(",", inputVectorLanes)}] and output lanes [{string.Join(",", outputVectorLanes)}].");
                }

                if (axis == inputShape.Length - 1)
                {
                    throw new NotSupportedException("PyNTT Gather does not support gathering along the vectorized value axis yet.");
                }

                valueVectorLaneCount = inputVectorLanes[0];
            }

            ValidateGatherShape("PyNTT Gather", inputShape, indexShape, outputShape, axis);
            _attrs["op"] = "gather";
            _attrs["axis"] = axis;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            var helperName = GetNextHelperName("gather_compute");
            WriteHelperTemplate(
                "triton/kernels/Gather.py.jinja",
                new PyNTTGatherTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferPointer(index),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTDTypeName(index.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetTritonDType(index.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    inputGlobalShape,
                    indexShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(index),
                    GetBufferStrides(output),
                    axis,
                    valueVectorLaneCount,
                    inputVectorLanes,
                    GetHierarchy(input),
                    GetBufferSplitAxes(input, inputGlobalShape.Length),
                    $"{input.Name}, {index.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitGatherReduceScatter(Nncase.TIR.NTT.GatherReduceScatter gatherReduceScatter, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2)
            {
                throw new NotSupportedException($"PyNTT GatherReduceScatter codegen expects 2 buffer operands, got {args.Count}.");
            }

            var input = GetBufferOperand(args[0], "PyNTT GatherReduceScatter input");
            var output = GetBufferOperand(args[1], "PyNTT GatherReduceScatter output");

            if (!gatherReduceScatter.InType.Placement.Hierarchy.SequenceEqual(gatherReduceScatter.OutType.Placement.Hierarchy))
            {
                throw new NotSupportedException("PyNTT GatherReduceScatter expects input and output placements to use the same hierarchy.");
            }

            SetComputeOp("reshard");
            var inputShape = GetBufferShape(input);
            var inputActiveShape = GetBufferActiveShape(input);
            var inputGlobalOffsets = GetBufferGlobalOffsets(input);
            var outputShape = GetBufferShape(output);
            var globalShape = GetRankedShapeDimensions(gatherReduceScatter.OutType.TensorType.Shape, "PyNTT GatherReduceScatter global shape")
                .Select(GetDimensionExpression)
                .ToArray();
            ValidateSameRank("PyNTT GatherReduceScatter input", inputShape, globalShape);
            ValidateSameRank("PyNTT GatherReduceScatter output", outputShape, globalShape);
            ValidateSameShape("PyNTT GatherReduceScatter global type", globalShape, GetRankedShapeDimensions(gatherReduceScatter.InType.TensorType.Shape, "PyNTT GatherReduceScatter input global shape").Select(GetDimensionExpression).ToArray());
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);
            if (inputVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT GatherReduceScatter expects matching input/output vector lanes, got input={inputVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            var inputScalarDType = GetPyNTTScalarDTypeName(input.ElemType);
            var outputScalarDType = GetPyNTTScalarDTypeName(output.ElemType);
            if (inputScalarDType != outputScalarDType)
            {
                throw new NotSupportedException($"PyNTT GatherReduceScatter expects matching scalar dtypes, got input={inputScalarDType}, output={outputScalarDType}.");
            }

            var hierarchy = gatherReduceScatter.OutType.Placement.Hierarchy.ToArray();
            if (gatherReduceScatter.InType.AxisPolicies.Any(policy => policy is SBPPartial) ||
                gatherReduceScatter.OutType.AxisPolicies.Any(policy => policy is SBPPartial))
            {
                throw new NotSupportedException("PyNTT GatherReduceScatter requires partial reduction metadata on DistributedType.Partial, not tensor-axis policies.");
            }

            if (gatherReduceScatter.OutType.Partial is not null)
            {
                throw new NotSupportedException("PyNTT GatherReduceScatter cannot materialize a partial output; partial reduction must be carried by the input type.");
            }

            var partialAxes = Array.Empty<int>();
            if (gatherReduceScatter.InType.Partial is { } partial)
            {
                if (partial.Op != ReduceOp.Sum)
                {
                    throw new NotSupportedException($"PyNTT GatherReduceScatter currently supports Sum partial reduction only, got {partial.Op}.");
                }

                partialAxes = partial.Axes.ToArray();
                if (partialAxes.Length == 0 || partialAxes.Distinct().Count() != partialAxes.Length)
                {
                    throw new NotSupportedException($"PyNTT GatherReduceScatter partial axes must be non-empty and unique, got [{string.Join(",", partialAxes)}].");
                }

                foreach (var axis in partialAxes)
                {
                    if (axis < 0 || axis >= hierarchy.Length)
                    {
                        throw new NotSupportedException($"PyNTT GatherReduceScatter partial reduce axis {axis} is outside hierarchy rank {hierarchy.Length}.");
                    }
                }
            }

            _attrs["op"] = "reshard";
            _attrs["dtype"] = outputScalarDType;
            _attrs["shape"] = globalShape;
            var outputRef = ResolveByteAddressedBufferRef(output);
            var partialInputRef = partialAxes.Length > 0
                ? ResolveByteAddressedBufferRef(input)
                : null;
            var inputSplitAxes = GetSplitAxes(gatherReduceScatter.InType);
            var conflictingAxes = inputSplitAxes.SelectMany(axes => axes).Intersect(partialAxes).Distinct().ToArray();
            if (conflictingAxes.Length > 0)
            {
                throw new NotSupportedException($"PyNTT GatherReduceScatter placement axes cannot be both split and partial: [{string.Join(",", conflictingAxes)}].");
            }

            var reductionGroupSize = partialAxes.Aggregate(1L, (size, axis) => checked(size * hierarchy[axis]));
            if (reductionGroupSize > 1 &&
                partialInputRef is { } reductionInputRef &&
                IsZeroOffset(reductionInputRef.PoolStrideBytes))
            {
                throw new NotSupportedException($"PyNTT GatherReduceScatter partial input {input.Name} requires distinct per-block storage.");
            }

            PyNTTReshardTemplateModel MakeModel(string helperName) => new(
                helperName,
                GetBufferScalarPointer(input),
                GetBufferScalarPointer(output),
                partialInputRef is null ? null : GetPooledByteAddressTemplateModel(partialInputRef),
                GetPooledByteAddressTemplateModel(outputRef),
                GetScalarElementSizeBytes(output.ElemType),
                outputScalarDType,
                GetScalarTritonDType(output.ElemType),
                globalShape,
                inputShape,
                inputActiveShape,
                inputGlobalOffsets,
                outputShape,
                GetBufferStrides(input),
                GetBufferStrides(output),
                inputVectorLaneCount,
                GetVectorLanes(input.ElemType),
                hierarchy,
                inputSplitAxes,
                partialAxes,
                GetSplitAxes(gatherReduceScatter.OutType),
                "tile_scatter",
                $"{input.Name} -> {output.Name}");

            var helperName = GetNextHelperName("reshard_tile_scatter");
            WriteHelperTemplate("triton/kernels/Reshard.py.jinja", MakeModel(helperName));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitPad(Nncase.TIR.NTT.Pad pad, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not Paddings paddings ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Pad codegen expects input buffer, static paddings, and output buffer.");
            }

            SetComputeOp("pad");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var pads = GetStaticPaddings(paddings, "PyNTT Pad");
            ValidatePadShape("PyNTT Pad", inputShape, outputShape, pads);
            _attrs["op"] = "pad";
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["pads"] = pads;
            _attrs["pad_value"] = pad.PadValue;
            var helperName = GetNextHelperName("pad_compute");
            WriteHelperTemplate(
                "triton/kernels/Pad.py.jinja",
                new PyNTTPadTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    pads,
                    pad.PadValue.ToString("R", CultureInfo.InvariantCulture),
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitScatterND(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer indices ||
                args[2] is not TIR.Buffer updates ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT ScatterND codegen expects input, indices, updates, and output TIR buffers.");
            }

            SetComputeOp("scatter_nd");
            var inputShape = GetBufferShape(input);
            var indicesShape = GetBufferShape(indices);
            var updatesShape = GetBufferShape(updates);
            var outputShape = GetBufferShape(output);
            ValidateScatterNDShape("PyNTT ScatterND", inputShape, indicesShape, updatesShape, outputShape);
            _attrs["op"] = "scatter_nd";
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            var helperName = GetNextHelperName("scatter_nd_compute");
            WriteHelperTemplate(
                "triton/kernels/ScatterND.py.jinja",
                new PyNTTScatterNDTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(indices),
                    GetBufferPointer(updates),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(indices.ElemType),
                    GetPyNTTDTypeName(updates.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(indices.ElemType),
                    GetTritonDType(updates.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    indicesShape,
                    updatesShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(indices),
                    GetBufferStrides(updates),
                    GetBufferStrides(output),
                    $"{input.Name}, {indices.Name}, {updates.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitSlice(Nncase.TIR.NTT.Slice slice, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer input ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Slice codegen expects input buffer, begins, ends, and output buffer.");
            }

            SetComputeOp("slice");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var starts = GetStaticRankedShape(args[1], "PyNTT Slice begins");
            var axes = slice.Axes.ToArray();
            var strides = slice.Strides.ToArray();
            var (normalizedStarts, normalizedStrides) = NormalizeSliceParameters(inputShape, starts, axes, strides, "PyNTT Slice");
            ValidateSameRank("PyNTT Slice", inputShape, outputShape);
            _attrs["op"] = "slice";
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["starts"] = normalizedStarts;
            _attrs["strides"] = normalizedStrides;
            var helperName = GetNextHelperName("slice_compute");
            WriteHelperTemplate(
                "triton/kernels/Slice.py.jinja",
                new PyNTTSliceTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    normalizedStarts,
                    normalizedStrides,
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitSwish(Nncase.TIR.NTT.Swish swish, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Swish codegen expects input and output TIR buffers.");
            }

            SetComputeOp("swish");
            var shape = GetBufferShape(output);
            var inputShape = GetBufferShape(input);
            ValidateSameShape("PyNTT Swish", inputShape, shape);
            ValidateMatchingVectorLanes("PyNTT Swish input/output", input.ElemType, output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            var beta = swish.Beta.ToString("R", CultureInfo.InvariantCulture);
            _attrs["op"] = "swish";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["beta"] = swish.Beta;
            var helperName = GetNextHelperName("swish_compute");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseUnary.py.jinja",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(input.ElemType),
                    GetVectorLanes(output.ElemType),
                    shape,
                    $"value0_f32 / (1.0 + tl.exp(-({beta}) * value0_f32))",
                    "swish",
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void VisitVectorizedBinary(Nncase.TIR.NTT.VectorizedBinary binary, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3)
            {
                throw new NotSupportedException("PyNTT VectorizedBinary codegen expects TIR buffer operands.");
            }

            var lhs = GetBufferOperand(args[0], "PyNTT VectorizedBinary lhs");
            var rhs = GetBufferOperand(args[1], "PyNTT VectorizedBinary rhs");
            var output = GetBufferOperand(args[2], "PyNTT VectorizedBinary output");

            SetComputeOp("binary");
            var shape = GetBufferShape(output);
            var lhsShape = GetBufferShape(lhs);
            var rhsShape = GetBufferShape(rhs);
            ValidateBroadcastable("PyNTT VectorizedBinary lhs", lhsShape, shape);
            ValidateBroadcastable("PyNTT VectorizedBinary rhs", rhsShape, shape);
            ValidateScalarOrMatchingVectorLanes("PyNTT VectorizedBinary lhs", lhs.ElemType, output.ElemType);
            ValidateScalarOrMatchingVectorLanes("PyNTT VectorizedBinary rhs", rhs.ElemType, output.ElemType);
            var lhsVectorLaneCount = GetVectorLaneElementCount(lhs.ElemType);
            var rhsVectorLaneCount = GetVectorLaneElementCount(rhs.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            _attrs["op"] = GetOpName(binary.BinaryOp);
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_binary");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseBinary.py.jinja",
                new PyNTTElementwiseBinaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(lhs),
                    GetBufferScalarPointer(rhs),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(lhs.ElemType),
                    GetPyNTTScalarDTypeName(rhs.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(lhs.ElemType),
                    GetScalarTritonDType(rhs.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    shape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
                    lhsVectorLaneCount,
                    rhsVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(lhs.ElemType),
                    GetVectorLanes(rhs.ElemType),
                    GetVectorLanes(output.ElemType),
                    shape,
                    GetBinaryExpression(binary.BinaryOp),
                    (string)_attrs["op"],
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
            MarkStoredOutput(output, "PyNTT VectorizedBinary");
        }

        private void VisitPack(Nncase.TIR.NTT.Pack pack, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Pack codegen expects input and output TIR buffers.");
            }

            SetComputeOp("pack");
            var inputShape = GetBufferActiveShape(input);
            var outputShape = GetBufferActiveShape(output);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var axes = NormalizeLayoutAxes(pack.Axes, inputShape.Length, "PyNTT Pack");
            var lanes = GetLayoutLanes(pack.Lanes, axes.Length, "PyNTT Pack");
            var outputGlobalShape = GetPackedShape(inputGlobalShape, axes, lanes);
            if (output.DistributedType is not null)
            {
                ValidatePackShape("PyNTT Pack global output", inputGlobalShape, GetBufferGlobalShape(output), axes, lanes);
            }

            SetBufferGlobalShapeMetadata(output, outputGlobalShape);
            ValidateSameRank("PyNTT Pack local", inputShape, outputShape);
            var inputLanes = GetVectorLanes(input.ElemType);
            var outputLanes = GetVectorLanes(output.ElemType);
            ValidateLanePrefix("PyNTT Pack output lanes", lanes.Concat(inputLanes).ToArray(), outputLanes);
            PropagatePackLayoutMetadata(input, output, axes, lanes);

            _attrs["op"] = "pack";
            _attrs["from_dtype"] = GetPyNTTDTypeName(input.ElemType);
            _attrs["to_dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axes"] = axes;
            _attrs["lanes"] = lanes;
            var helperName = GetNextHelperName("pack_compute");
            WriteHelperTemplate(
                "triton/kernels/VectorLayout.py.jinja",
                new PyNTTVectorLayoutTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputLanes,
                    outputLanes,
                    axes,
                    lanes,
                    true,
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void VisitUnpack(Nncase.TIR.NTT.Unpack unpack, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Unpack codegen expects input and output TIR buffers.");
            }

            SetComputeOp("unpack");
            var inputShape = GetBufferActiveShape(input);
            var outputShape = GetBufferActiveShape(output);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var axes = NormalizeLayoutAxes(unpack.Axes, outputShape.Length, "PyNTT Unpack");
            var lanes = GetLayoutLanes(unpack.Lanes, axes.Length, "PyNTT Unpack");
            var outputGlobalShape = GetUnpackedShape(inputGlobalShape, axes, lanes);
            if (output.DistributedType is not null)
            {
                ValidateUnpackShape("PyNTT Unpack global output", inputGlobalShape, GetBufferGlobalShape(output), axes, lanes);
            }

            SetBufferGlobalShapeMetadata(output, outputGlobalShape);
            ValidateSameRank("PyNTT Unpack local", inputShape, outputShape);
            var inputLanes = GetVectorLanes(input.ElemType);
            var outputLanes = GetVectorLanes(output.ElemType);
            ValidateLanePrefix("PyNTT Unpack input lanes", lanes.Concat(outputLanes).ToArray(), inputLanes);
            PropagateUnpackLayoutMetadata(input, output, axes, lanes);

            _attrs["op"] = "unpack";
            _attrs["from_dtype"] = GetPyNTTDTypeName(input.ElemType);
            _attrs["to_dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axes"] = axes;
            _attrs["lanes"] = lanes;
            var helperName = GetNextHelperName("unpack_compute");
            WriteHelperTemplate(
                "triton/kernels/VectorLayout.py.jinja",
                new PyNTTVectorLayoutTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputLanes,
                    outputLanes,
                    axes,
                    lanes,
                    false,
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void VisitCast(Nncase.TIR.NTT.Cast cast, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2)
            {
                throw new NotSupportedException("PyNTT Cast codegen expects TIR buffer operands.");
            }

            var input = GetBufferOperand(args[0], "PyNTT Cast input");
            var output = GetBufferOperand(args[1], "PyNTT Cast output");
            if (args.Count > 2 && args[2] is not None)
            {
                throw new NotSupportedException("PyNTT Cast codegen does not support post ops yet.");
            }

            SetComputeOp("cast");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT Cast input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Cast output");
            var vectorizedAxes = cast.VectorizeAxes.ToArray();
            if (vectorizedAxes.Length > 1)
            {
                throw new NotSupportedException($"PyNTT Cast supports at most one vectorized axis, got [{string.Join(",", vectorizedAxes)}].");
            }

            if ((inputVectorLaneCount != 1 || outputVectorLaneCount != 1) && vectorizedAxes.Length == 0)
            {
                throw new NotSupportedException("PyNTT Cast with vector dtype expects one vectorized axis.");
            }

            var logicalInputShape = vectorizedAxes.Length == 0
                ? inputShape
                : GetLogicalVectorShape(inputShape, vectorizedAxes[0], inputVectorLaneCount);
            var logicalOutputShape = vectorizedAxes.Length == 0
                ? outputShape
                : GetLogicalVectorShape(outputShape, vectorizedAxes[0], outputVectorLaneCount);
            ValidateSameShape("PyNTT Cast", logicalInputShape, logicalOutputShape);
            _attrs["op"] = "cast";
            _attrs["from_dtype"] = GetPyNTTScalarDTypeName(input.ElemType);
            _attrs["to_dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["cast_mode"] = GetCastModeName(cast.CastMode);
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = logicalOutputShape;
            var helperName = GetNextHelperName("elementwise_cast");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseCast.py.jinja",
                new PyNTTElementwiseCastTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(input.ElemType),
                    GetVectorLanes(output.ElemType),
                    vectorizedAxes,
                    logicalOutputShape,
                    GetCastExpression(cast.CastMode, output.ElemType),
                    (string)_attrs["cast_mode"],
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
            MarkStoredOutput(output, "PyNTT Cast");
        }

        private void VisitWhere(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer cond ||
                args[1] is not TIR.Buffer trueValue ||
                args[2] is not TIR.Buffer falseValue ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Where codegen expects TIR buffer operands.");
            }

            SetComputeOp("where");
            var shape = GetBufferShape(output);
            var condShape = GetBufferShape(cond);
            var trueShape = GetBufferShape(trueValue);
            var falseShape = GetBufferShape(falseValue);
            var condVectorLaneCount = GetSingleVectorLaneCount(cond.ElemType, "PyNTT Where cond");
            var trueVectorLaneCount = GetSingleVectorLaneCount(trueValue.ElemType, "PyNTT Where true value");
            var falseVectorLaneCount = GetSingleVectorLaneCount(falseValue.ElemType, "PyNTT Where false value");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Where output");
            var valueLaneCounts = new[] { trueVectorLaneCount, falseVectorLaneCount, outputVectorLaneCount }
                .Where(lane => lane > 1)
                .Distinct()
                .ToArray();
            if (valueLaneCounts.Length > 1)
            {
                throw new NotSupportedException($"PyNTT Where expects matching value/output vector lanes, got true={trueVectorLaneCount}, false={falseVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            var logicalShape = GetLogicalVectorShape(shape, outputVectorLaneCount);
            ValidateBroadcastable("PyNTT Where cond", GetLogicalVectorShape(condShape, condVectorLaneCount), logicalShape);
            ValidateBroadcastable("PyNTT Where true value", GetLogicalVectorShape(trueShape, trueVectorLaneCount), logicalShape);
            ValidateBroadcastable("PyNTT Where false value", GetLogicalVectorShape(falseShape, falseVectorLaneCount), logicalShape);
            _attrs["op"] = "where";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = logicalShape;
            var helperName = GetNextHelperName("elementwise_where");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseWhere.py.jinja",
                new PyNTTElementwiseWhereTemplateModel(
                    helperName,
                    GetBufferScalarPointer(cond),
                    GetBufferScalarPointer(trueValue),
                    GetBufferScalarPointer(falseValue),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(cond.ElemType),
                    GetPyNTTScalarDTypeName(trueValue.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(cond.ElemType),
                    GetScalarTritonDType(trueValue.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    condShape,
                    trueShape,
                    falseShape,
                    shape,
                    GetBufferStrides(cond),
                    GetBufferStrides(trueValue),
                    GetBufferStrides(falseValue),
                    GetBufferStrides(output),
                    condVectorLaneCount,
                    trueVectorLaneCount,
                    falseVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(cond.ElemType),
                    GetVectorLanes(trueValue.ElemType),
                    GetVectorLanes(falseValue.ElemType),
                    GetVectorLanes(output.ElemType),
                    logicalShape,
                    $"{cond.Name}, {trueValue.Name}, {falseValue.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitClamp(Nncase.TIR.NTT.Clamp clamp, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Clamp codegen expects input and output TIR buffers.");
            }

            SetComputeOp("clamp");
            var shape = GetBufferShape(output);
            var inputShape = GetBufferShape(input);
            ValidateSameShape("PyNTT Clamp", inputShape, shape);
            ValidateMatchingVectorLanes("PyNTT Clamp input/output", input.ElemType, output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            _attrs["op"] = "clamp";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["min"] = clamp.Min;
            _attrs["max"] = clamp.Max;
            var helperName = GetNextHelperName("elementwise_clamp");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseUnary.py.jinja",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(input.ElemType),
                    GetVectorLanes(output.ElemType),
                    shape,
                    GetClampExpression(clamp.Min, clamp.Max),
                    "clamp",
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void VisitCompare(Nncase.TIR.NTT.Compare compare, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Compare codegen expects lhs, rhs, and output TIR buffers.");
            }

            SetComputeOp("compare");
            var shape = GetBufferShape(output);
            var lhsShape = GetBufferShape(lhs);
            var rhsShape = GetBufferShape(rhs);
            ValidateBroadcastable("PyNTT Compare lhs", lhsShape, shape);
            ValidateBroadcastable("PyNTT Compare rhs", rhsShape, shape);
            ValidateScalarOrMatchingVectorLanes("PyNTT Compare lhs", lhs.ElemType, output.ElemType);
            ValidateScalarOrMatchingVectorLanes("PyNTT Compare rhs", rhs.ElemType, output.ElemType);
            var lhsVectorLaneCount = GetVectorLaneElementCount(lhs.ElemType);
            var rhsVectorLaneCount = GetVectorLaneElementCount(rhs.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            _attrs["op"] = GetCompareOpName(compare.CompareOp);
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_compare");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseBinary.py.jinja",
                new PyNTTElementwiseBinaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(lhs),
                    GetBufferScalarPointer(rhs),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(lhs.ElemType),
                    GetPyNTTScalarDTypeName(rhs.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(lhs.ElemType),
                    GetScalarTritonDType(rhs.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    shape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
                    lhsVectorLaneCount,
                    rhsVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(lhs.ElemType),
                    GetVectorLanes(rhs.ElemType),
                    GetVectorLanes(output.ElemType),
                    shape,
                    GetCompareExpression(compare.CompareOp),
                    (string)_attrs["op"],
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void VisitConcat(Nncase.TIR.NTT.Concat concat, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[^1] is not TIR.Buffer output ||
                args.Take(args.Count - 1).Any(arg => arg is not TIR.Buffer))
            {
                throw new NotSupportedException("PyNTT Concat codegen expects one or more input TIR buffers and one output TIR buffer.");
            }

            SetComputeOp("concat");
            var inputs = args.Take(args.Count - 1).Cast<TIR.Buffer>().ToArray();
            var outputShape = GetBufferShape(output);
            if (outputShape.Length == 0)
            {
                throw new NotSupportedException("PyNTT Concat codegen requires ranked tensor buffers.");
            }

            var axis = NormalizeAxis(concat.Axis, outputShape.Length, "PyNTT Concat");
            var inputShapes = inputs.Select(GetBufferShape).ToArray();
            var inputStrides = inputs.Select(GetBufferStrides).ToArray();
            var concatAxisExtent = PyNTTDimExpression.Zero;
            for (var inputIndex = 0; inputIndex < inputShapes.Length; inputIndex++)
            {
                var inputShape = inputShapes[inputIndex];
                if (inputShape.Length != outputShape.Length)
                {
                    throw new NotSupportedException($"PyNTT Concat input{inputIndex} rank {inputShape.Length} does not match output rank {outputShape.Length}.");
                }

                for (var dim = 0; dim < outputShape.Length; dim++)
                {
                    if (dim != axis && !SameDim(inputShape[dim], outputShape[dim]))
                    {
                        throw new NotSupportedException($"PyNTT Concat input{inputIndex} dim {dim}={inputShape[dim]} does not match output dim {outputShape[dim]}.");
                    }
                }

                concatAxisExtent = AddDims(concatAxisExtent, inputShape[axis]);
            }

            if (!SameDim(concatAxisExtent, outputShape[axis]))
            {
                throw new NotSupportedException($"PyNTT Concat input axis extent sum {concatAxisExtent} does not match output axis extent {outputShape[axis]}.");
            }

            _attrs["op"] = "concat";
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axis"] = axis;
            _attrs["input_count"] = inputs.Length;
            var helperName = GetNextHelperName("concat_compute");
            WriteHelperTemplate(
                "triton/kernels/Concat.py.jinja",
                new PyNTTConcatTemplateModel(
                    helperName,
                    inputs.Select(input => GetBufferPointer(input)).ToArray(),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShapes,
                    inputStrides,
                    outputShape,
                    GetBufferStrides(output),
                    axis,
                    $"{string.Join(", ", inputs.Select(input => input.Name))} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitConv2D(Nncase.TIR.NTT.Conv2D conv2D, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer weights ||
                args[2] is not TIR.Buffer bias ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Conv2D codegen expects input, weights, bias, and output TIR buffers.");
            }

            if (conv2D.PadMode != PadMode.Constant)
            {
                throw new NotSupportedException($"PyNTT Conv2D currently supports constant padding only, got {conv2D.PadMode}.");
            }

            SetComputeOp("conv2d");
            var inputShape = GetBufferShape(input);
            var weightsShape = GetBufferShape(weights);
            var biasShape = GetBufferShape(bias);
            var outputShape = GetBufferShape(output);
            ValidateRank("PyNTT Conv2D input", inputShape, 4);
            ValidateRank("PyNTT Conv2D weights", weightsShape, 4);
            ValidateRank("PyNTT Conv2D bias", biasShape, 1);
            ValidateRank("PyNTT Conv2D output", outputShape, 4);

            var stride = conv2D.Stride.ToArray();
            var padding = conv2D.Padding.ToArray();
            var dilation = conv2D.Dilation.ToArray();
            if (stride.Length != 2 || padding.Length != 4 || dilation.Length != 2)
            {
                throw new NotSupportedException($"PyNTT Conv2D expects stride rank 2, padding rank 4, dilation rank 2; got stride={stride.Length}, padding={padding.Length}, dilation={dilation.Length}.");
            }

            if (conv2D.Groups <= 0)
            {
                throw new NotSupportedException($"PyNTT Conv2D requires positive groups, got {conv2D.Groups}.");
            }

            if (!SameDim(outputShape[1], weightsShape[0]) || !SameDim(biasShape[0], outputShape[1]))
            {
                throw new NotSupportedException($"PyNTT Conv2D expects output channels, weights, and bias to match; got output={outputShape[1]}, weights={weightsShape[0]}, bias={biasShape[0]}.");
            }

            var weightsInputChannels = MultiplyDim(weightsShape[1], conv2D.Groups);
            if (!SameDim(inputShape[1], weightsInputChannels))
            {
                throw new NotSupportedException($"PyNTT Conv2D expects input channels {inputShape[1]} to equal weights input channels {weightsShape[1]} * groups {conv2D.Groups}.");
            }

            var outputChannels = RequireFixedDim(outputShape[1], "PyNTT Conv2D output channels");
            if (outputChannels % conv2D.Groups != 0)
            {
                throw new NotSupportedException($"PyNTT Conv2D expects output channels {outputShape[1]} to be divisible by groups {conv2D.Groups}.");
            }

            _attrs["op"] = "conv2d";
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["stride"] = stride;
            _attrs["padding"] = padding;
            _attrs["dilation"] = dilation;
            _attrs["groups"] = conv2D.Groups;
            var helperName = GetNextHelperName("conv2d_compute");
            WriteHelperTemplate(
                "triton/kernels/Conv2D.py.jinja",
                new PyNTTConv2DTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(weights),
                    GetBufferPointer(bias),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(weights.ElemType),
                    GetPyNTTDTypeName(bias.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(weights.ElemType),
                    GetTritonDType(bias.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    weightsShape,
                    biasShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(weights),
                    GetBufferStrides(bias),
                    GetBufferStrides(output),
                    stride,
                    padding,
                    dilation,
                    conv2D.Groups,
                    $"{input.Name}, {weights.Name}, {bias.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitTranspose(Nncase.TIR.NTT.Transpose transpose, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Transpose codegen expects input and output TIR buffers.");
            }

            SetComputeOp("transpose");
            var inputShape = GetBufferActiveShape(input);
            var outputShape = GetBufferActiveShape(output);
            var perm = transpose.Perm.ToArray();
            if (perm.Length != inputShape.Length || outputShape.Length != inputShape.Length)
            {
                throw new NotSupportedException($"PyNTT Transpose expects input/output/perm ranks to match, got input={inputShape.Length}, output={outputShape.Length}, perm={perm.Length}.");
            }

            var sortedPerm = perm.OrderBy(axis => axis).ToArray();
            if (!sortedPerm.SequenceEqual(Enumerable.Range(0, perm.Length)))
            {
                throw new NotSupportedException($"PyNTT Transpose expects a valid permutation, got [{string.Join(",", perm)}].");
            }

            for (var outputAxis = 0; outputAxis < outputShape.Length; outputAxis++)
            {
                var inputAxis = perm[outputAxis];
                if (!SameDim(outputShape[outputAxis], inputShape[inputAxis]))
                {
                    throw new NotSupportedException($"PyNTT Transpose output axis {outputAxis} shape {outputShape[outputAxis]} does not match input axis {inputAxis} shape {inputShape[inputAxis]}.");
                }
            }

            _attrs["op"] = "transpose";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["perm"] = perm;
            ValidateMatchingVectorLanes("PyNTT Transpose input/output", input.ElemType, output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);
            PropagateTransposeLayoutMetadata(input, output, perm);

            var helperName = GetNextHelperName("transpose_compute");
            WriteHelperTemplate(
                "triton/kernels/Transpose.py.jinja",
                new PyNTTTransposeTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(input.ElemType),
                    GetVectorLanes(output.ElemType),
                    perm,
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void PropagatePackLayoutMetadata(TIR.Buffer input, TIR.Buffer output, IReadOnlyList<int> axes, IReadOnlyList<int> lanes)
        {
            ValidateSameRank("PyNTT Pack layout metadata", input.Rank, output.Rank);
            var outputOffsets = GetBufferGlobalOffsets(input).ToArray();
            for (var axis = 0; axis < outputOffsets.Length; axis++)
            {
                var laneProduct = GetLayoutAxisLaneProduct(axes, lanes, axis);
                if (laneProduct > 1)
                {
                    outputOffsets[axis] = FloorDivDim(outputOffsets[axis], laneProduct);
                }
            }

            SetBufferLayoutMetadata(output, outputOffsets, GetBufferSourceSplitAxes(input, input.Rank));
        }

        private void PropagateUnpackLayoutMetadata(TIR.Buffer input, TIR.Buffer output, IReadOnlyList<int> axes, IReadOnlyList<int> lanes)
        {
            ValidateSameRank("PyNTT Unpack layout metadata", input.Rank, output.Rank);
            var outputOffsets = GetBufferGlobalOffsets(input).ToArray();
            for (var axis = 0; axis < outputOffsets.Length; axis++)
            {
                var laneProduct = GetLayoutAxisLaneProduct(axes, lanes, axis);
                if (laneProduct > 1)
                {
                    outputOffsets[axis] = MultiplyDim(outputOffsets[axis], laneProduct);
                }
            }

            SetBufferLayoutMetadata(output, outputOffsets, GetBufferSourceSplitAxes(input, input.Rank));
        }

        private void PropagateTransposeLayoutMetadata(TIR.Buffer input, TIR.Buffer output, IReadOnlyList<int> perm)
        {
            var inputOffsets = GetBufferGlobalOffsets(input);
            var inputSourceSplitAxes = GetBufferSourceSplitAxes(input, input.Rank);
            SetBufferLayoutMetadata(
                output,
                perm.Select(axis => inputOffsets[axis]).ToArray(),
                perm.Select(axis => inputSourceSplitAxes[axis].ToArray()).ToArray());
        }

        private void SetBufferLayoutMetadata(TIR.Buffer buffer, PyNTTDimExpression[] globalOffsets, IReadOnlyList<int[]> sourceSplitAxes)
        {
            _bufferGlobalOffsetOverrides[buffer] = globalOffsets;
            _bufferSourceSplitAxesOverrides[buffer] = CloneSplitAxes(sourceSplitAxes);

            if (!_bufferViewSourceByBuffer.TryGetValue(buffer, out var viewSource) || !IsFullBufferView(viewSource))
            {
                return;
            }

            _bufferGlobalOffsetOverrides[viewSource.Source] = globalOffsets;
            _bufferSourceSplitAxesOverrides[viewSource.Source] = CloneSplitAxes(sourceSplitAxes);
        }

        private void SetBufferGlobalShapeMetadata(TIR.Buffer buffer, PyNTTDimExpression[] globalShape)
        {
            _bufferGlobalShapeOverrides[buffer] = globalShape;
            if (_bufferViewSourceByBuffer.TryGetValue(buffer, out var viewSource) && IsFullBufferView(viewSource))
            {
                _bufferGlobalShapeOverrides[viewSource.Source] = globalShape;
            }
        }

        private bool IsFullBufferView(BufferViewSource viewSource)
        {
            if (viewSource.Offsets.Any(offset => offset.FixedValue != 0))
            {
                return false;
            }

            var sourceShape = GetBufferActiveShape(viewSource.Source);
            return sourceShape.Length == viewSource.Shape.Length &&
                sourceShape.Zip(viewSource.Shape).All(pair => SameDim(pair.First, pair.Second));
        }

        private void VisitMatmul(Nncase.TIR.NTT.Matmul matmul, IReadOnlyList<BaseExpr> args)
        {
            if (matmul.FusedReduce)
            {
                throw new NotSupportedException("PyNTT Matmul codegen does not support fused reduce yet.");
            }

            if (args.Count > 5 && args[5] is not None)
            {
                throw new NotSupportedException("PyNTT Matmul codegen does not support extra workload operands yet.");
            }

            VisitMatmulLike(
                "PyNTT Matmul",
                matmul.LhsVectorizedAxes,
                matmul.RhsVectorizedAxes,
                matmul.TransposeA,
                matmul.TransposeB,
                args);
        }

        private void VisitPackedMatmul(Nncase.TIR.NTT.PackedMatMul matmul, IReadOnlyList<BaseExpr> args)
        {
            if (matmul.FusedReduce)
            {
                throw new NotSupportedException("PyNTT PackedMatMul codegen does not support fused reduce yet.");
            }

            if (args.Count < 5 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT PackedMatMul codegen expects lhs, rhs, and output TIR buffers.");
            }

            var loadCExpression = GetScalarBoolExpression(args[3], "PyNTT PackedMatMul loadC");

            SetComputeOp("matmul");
            var lhsShape = GetBufferActiveShape(lhs);
            var rhsShape = GetBufferActiveShape(rhs);
            var outputShape = GetBufferActiveShape(output);
            ValidateMinimumRank("PyNTT PackedMatMul lhs", lhsShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMul rhs", rhsShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMul output", outputShape, 2);

            var lhsVectorLanes = GetVectorLanes(lhs.ElemType);
            var rhsVectorLanes = GetVectorLanes(rhs.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            if (lhsVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"PyNTT PackedMatMul expects scalar lhs operands, got lhs lanes [{string.Join(",", lhsVectorLanes)}].");
            }

            if (rhsVectorLanes.Length != 2)
            {
                throw new NotSupportedException($"PyNTT PackedMatMul expects RHS vector lanes [Nr,lane], got [{string.Join(",", rhsVectorLanes)}].");
            }

            if (outputVectorLanes.Length != 2 || outputVectorLanes[0] != rhsVectorLanes[0] || outputVectorLanes[1] != rhsVectorLanes[1])
            {
                throw new NotSupportedException($"PyNTT PackedMatMul expects output lanes [{rhsVectorLanes[0]},{rhsVectorLanes[1]}], got [{string.Join(",", outputVectorLanes)}].");
            }

            var nPackedLaneCount = rhsVectorLanes[0];
            var nVectorLaneCount = rhsVectorLanes[1];
            var rhsNScalarLaneCount = checked(nPackedLaneCount * nVectorLaneCount);
            var lhsK = lhsShape[^1];
            var rhsK = rhsShape[^1];
            var lhsM = lhsShape[^2];
            var rhsNOuter = rhsShape[^2];
            if (!SameDim(lhsK, rhsK) ||
                !SameDim(outputShape[^2], lhsM) ||
                !SameDim(outputShape[^1], rhsNOuter))
            {
                throw new NotSupportedException($"PyNTT PackedMatMul expects lhs=[...,M,K], rhs=[...,N,K]<Nr,lane>, output=[...,M,N]<Nr,lane>, got lhs=[{ShapeText(lhsShape)}], rhs=[{ShapeText(rhsShape)}], output=[{ShapeText(outputShape)}].");
            }

            ValidateBroadcastable("PyNTT PackedMatMul lhs batch", lhsShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedMatMul rhs batch", rhsShape[..^2], outputShape[..^2]);
            _ = Nncase.IR.NTT.VectorizedMatMul.GetDimInfo(false, true, lhsShape.Length, rhsShape.Length);
            var scale = GetScalarFloat(args[4], "PyNTT PackedMatMul scale", 1.0f);
            _attrs["op"] = "packed_matmul";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["n_pack_lane"] = nPackedLaneCount;
            _attrs["n_lane"] = nVectorLaneCount;
            _attrs["n_scalar_lane"] = rhsNScalarLaneCount;
            _attrs["scale"] = scale;
            var useGemv = IsGemvMatmul(outputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "gemv_compute" : "matmul_compute");
            var templateModel = new PyNTTMatmulTemplateModel(
                helperName,
                GetBufferScalarPointer(lhs),
                GetBufferScalarPointer(rhs),
                GetBufferScalarPointer(output),
                GetPyNTTScalarDTypeName(lhs.ElemType),
                GetPyNTTScalarDTypeName(rhs.ElemType),
                GetPyNTTScalarDTypeName(output.ElemType),
                GetScalarTritonDType(lhs.ElemType),
                GetScalarTritonDType(rhs.ElemType),
                GetScalarTritonDType(output.ElemType),
                lhsShape,
                rhsShape,
                outputShape,
                GetBufferStrides(lhs),
                GetBufferStrides(rhs),
                GetBufferStrides(output),
                false,
                true,
                GetHierarchy(output),
                nVectorLaneCount,
                outputVectorLanes[1],
                scale.ToString("R", CultureInfo.InvariantCulture),
                $"{lhs.Name}, {rhs.Name} -> {output.Name}")
            {
                RhsNPackedLaneCount = nPackedLaneCount,
                OutputNPackedLaneCount = outputVectorLanes[0],
                LoadCExpression = loadCExpression,
            };

            if (_currentReductionState is { } reductionState)
            {
                var logicalOutputShape = outputShape.ToArray();
                logicalOutputShape[^1] = MultiplyDim(logicalOutputShape[^1], rhsNScalarLaneCount);
                EmitMatmulReductionUpdate(
                    reductionState,
                    ReductionKernelKind.PackedMatmul,
                    templateModel,
                    useGemv ? "triton/kernels/Gemv.py.jinja" : "triton/kernels/Matmul.py.jinja",
                    logicalOutputShape,
                    useGemv);
                return;
            }

            WriteHelperTemplate(
                useGemv ? "triton/kernels/Gemv.py.jinja" : "triton/kernels/Matmul.py.jinja",
                templateModel);
            WriteHelperInvocation(helperName);
        }

        private void VisitQKVParallelLinear(Nncase.TIR.NTT.QKVParallelLinear qkv, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 16 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer qWeight ||
                args[2] is not TIR.Buffer kWeight ||
                args[3] is not TIR.Buffer vWeight ||
                args[13] is not TIR.Buffer qOutput ||
                args[14] is not TIR.Buffer kOutput ||
                args[15] is not TIR.Buffer vOutput)
            {
                throw new NotSupportedException("PyNTT QKVParallelLinear codegen expects input, q/k/v weights, optional q/k/v bias and scale values, and q/k/v output TIR buffers.");
            }

            TIR.Buffer? GetOptionalBuffer(int index, string name) => args[index] switch
            {
                TIR.Buffer buffer => buffer,
                None => null,
                _ => throw new NotSupportedException($"PyNTT QKVParallelLinear expects {name} to be a TIR buffer or None, got {args[index].GetType().Name}."),
            };

            if (args.Skip(7).Take(6).Any(arg => arg is not None))
            {
                throw new NotSupportedException("PyNTT QKVParallelLinear codegen currently supports only None input/weight scales.");
            }

            var qBias = GetOptionalBuffer(4, "q bias");
            var kBias = GetOptionalBuffer(5, "k bias");
            var vBias = GetOptionalBuffer(6, "v bias");
            SetComputeOp("qkv_parallel_linear");
            var inputShape = GetBufferActiveShape(input);
            var qWeightShape = GetBufferActiveShape(qWeight);
            var kWeightShape = GetBufferActiveShape(kWeight);
            var vWeightShape = GetBufferActiveShape(vWeight);
            var qBiasShape = qBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(qBias);
            var kBiasShape = kBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(kBias);
            var vBiasShape = vBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(vBias);
            var qOutputShape = GetBufferActiveShape(qOutput);
            var kOutputShape = GetBufferActiveShape(kOutput);
            var vOutputShape = GetBufferActiveShape(vOutput);
            ValidateMinimumRank("PyNTT QKVParallelLinear input", inputShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear q weight", qWeightShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear k weight", kWeightShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear v weight", vWeightShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear q output", qOutputShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear k output", kOutputShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear v output", vOutputShape, 2);

            foreach (var (name, buffer) in new[]
            {
                ("input", input),
                ("q weight", qWeight),
                ("k weight", kWeight),
                ("v weight", vWeight),
                ("q output", qOutput),
                ("k output", kOutput),
                ("v output", vOutput),
            })
            {
                var lanes = GetVectorLanes(buffer.ElemType);
                if (lanes.Length != 0)
                {
                    throw new NotSupportedException($"PyNTT QKVParallelLinear currently expects scalar {name} operands, got lanes [{string.Join(",", lanes)}].");
                }
            }

            ValidateProjectionShape("q", inputShape, qWeightShape, qOutputShape);
            ValidateProjectionShape("k", inputShape, kWeightShape, kOutputShape);
            ValidateProjectionShape("v", inputShape, vWeightShape, vOutputShape);
            ValidateBiasShape("q", qBiasShape, qOutputShape);
            ValidateBiasShape("k", kBiasShape, kOutputShape);
            ValidateBiasShape("v", vBiasShape, vOutputShape);
            ValidateBroadcastable("PyNTT QKVParallelLinear input/q output batch", inputShape[..^2], qOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear input/k output batch", inputShape[..^2], kOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear input/v output batch", inputShape[..^2], vOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear q weight/q output batch", qWeightShape[..^2], qOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear k weight/k output batch", kWeightShape[..^2], kOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear v weight/v output batch", vWeightShape[..^2], vOutputShape[..^2]);

            if (input.ElemType != qWeight.ElemType || qWeight.ElemType != kWeight.ElemType || kWeight.ElemType != vWeight.ElemType)
            {
                throw new NotSupportedException($"PyNTT QKVParallelLinear expects input and q/k/v weights to have the same dtype, got input={input.ElemType}, q={qWeight.ElemType}, k={kWeight.ElemType}, v={vWeight.ElemType}.");
            }

            if (qOutput.ElemType != kOutput.ElemType || kOutput.ElemType != vOutput.ElemType)
            {
                throw new NotSupportedException($"PyNTT QKVParallelLinear expects q/k/v outputs to have the same dtype, got q={qOutput.ElemType}, k={kOutput.ElemType}, v={vOutput.ElemType}.");
            }

            var biasDType = qBias?.ElemType ?? kBias?.ElemType ?? vBias?.ElemType ?? qOutput.ElemType;
            _attrs["op"] = "qkv_parallel_linear";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(qOutput.ElemType);
            _attrs["has_bias"] = qBias is not null || kBias is not null || vBias is not null;
            _attrs["q_shape"] = qOutputShape;
            _attrs["k_shape"] = kOutputShape;
            _attrs["v_shape"] = vOutputShape;
            _attrs["num_heads"] = qkv.NumHeads;
            _attrs["num_kv_heads"] = qkv.NumKvHeads;
            var useGemv = IsGemvMatmul(qOutputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "qkv_gemv_compute" : "qkv_matmul_compute");
            var templateModel = new PyNTTQKVParallelLinearTemplateModel(
                helperName,
                GetBufferScalarPointer(input),
                GetBufferScalarPointer(qWeight),
                GetBufferScalarPointer(kWeight),
                GetBufferScalarPointer(vWeight),
                qBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(qBias),
                kBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(kBias),
                vBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(vBias),
                GetBufferScalarPointer(qOutput),
                GetBufferScalarPointer(kOutput),
                GetBufferScalarPointer(vOutput),
                qBias is not null,
                kBias is not null,
                vBias is not null,
                GetPyNTTScalarDTypeName(input.ElemType),
                GetPyNTTScalarDTypeName(qWeight.ElemType),
                GetPyNTTScalarDTypeName(biasDType),
                GetPyNTTScalarDTypeName(qOutput.ElemType),
                GetScalarTritonDType(input.ElemType),
                GetScalarTritonDType(qWeight.ElemType),
                GetScalarTritonDType(biasDType),
                GetScalarTritonDType(qOutput.ElemType),
                inputShape,
                qWeightShape,
                kWeightShape,
                vWeightShape,
                qBiasShape,
                kBiasShape,
                vBiasShape,
                qOutputShape,
                kOutputShape,
                vOutputShape,
                GetBufferStrides(input),
                GetBufferStrides(qWeight),
                GetBufferStrides(kWeight),
                GetBufferStrides(vWeight),
                qBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(qBias),
                kBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(kBias),
                vBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(vBias),
                GetBufferStrides(qOutput),
                GetBufferStrides(kOutput),
                GetBufferStrides(vOutput),
                GetHierarchy(qOutput),
                $"{input.Name}, ({qWeight.Name}, {kWeight.Name}, {vWeight.Name}) -> ({qOutput.Name}, {kOutput.Name}, {vOutput.Name})");

            if (_currentReductionState is { } reductionState)
            {
                EmitQKVParallelLinearReductionUpdate(
                    reductionState,
                    templateModel,
                    "triton/kernels/QKVParallelLinear.py.jinja",
                    qOutputShape,
                    kOutputShape,
                    vOutputShape,
                    useGemv);
                return;
            }

            WriteHelperTemplate("triton/kernels/QKVParallelLinear.py.jinja", templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitPackedQKVParallelLinear(Nncase.TIR.NTT.PackedQKVParallelLinear qkv, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 16 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer qWeight ||
                args[2] is not TIR.Buffer kWeight ||
                args[3] is not TIR.Buffer vWeight ||
                args[13] is not TIR.Buffer qOutput ||
                args[14] is not TIR.Buffer kOutput ||
                args[15] is not TIR.Buffer vOutput)
            {
                throw new NotSupportedException("PyNTT PackedQKVParallelLinear codegen expects input, packed q/k/v weights, optional packed q/k/v bias and scale values, and packed q/k/v output TIR buffers.");
            }

            TIR.Buffer? GetOptionalBuffer(int index, string name) => args[index] switch
            {
                TIR.Buffer buffer => buffer,
                None => null,
                _ => throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects {name} to be a TIR buffer or None, got {args[index].GetType().Name}."),
            };

            if (args.Skip(7).Take(6).Any(arg => arg is not None))
            {
                throw new NotSupportedException("PyNTT PackedQKVParallelLinear codegen currently supports only None input/weight scales.");
            }

            var qBias = GetOptionalBuffer(4, "q bias");
            var kBias = GetOptionalBuffer(5, "k bias");
            var vBias = GetOptionalBuffer(6, "v bias");
            SetComputeOp("packed_qkv_parallel_linear");
            var inputShape = GetBufferActiveShape(input);
            var qWeightShape = GetBufferActiveShape(qWeight);
            var kWeightShape = GetBufferActiveShape(kWeight);
            var vWeightShape = GetBufferActiveShape(vWeight);
            var qBiasShape = qBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(qBias);
            var kBiasShape = kBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(kBias);
            var vBiasShape = vBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(vBias);
            var qOutputShape = GetBufferActiveShape(qOutput);
            var kOutputShape = GetBufferActiveShape(kOutput);
            var vOutputShape = GetBufferActiveShape(vOutput);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear input", inputShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear q weight", qWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear k weight", kWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear v weight", vWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear q output", qOutputShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear k output", kOutputShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear v output", vOutputShape, 2);

            var inputVectorLanes = GetVectorLanes(input.ElemType);
            if (inputVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects scalar input operands, got lanes [{string.Join(",", inputVectorLanes)}].");
            }

            var qWeightLanes = GetVectorLanes(qWeight.ElemType);
            var kWeightLanes = GetVectorLanes(kWeight.ElemType);
            var vWeightLanes = GetVectorLanes(vWeight.ElemType);
            var qOutputLanes = GetVectorLanes(qOutput.ElemType);
            var kOutputLanes = GetVectorLanes(kOutput.ElemType);
            var vOutputLanes = GetVectorLanes(vOutput.ElemType);
            ValidatePackedQKVLanes("q", qWeightLanes, qOutputLanes);
            ValidatePackedQKVLanes("k", kWeightLanes, kOutputLanes);
            ValidatePackedQKVLanes("v", vWeightLanes, vOutputLanes);
            if (!qWeightLanes.SequenceEqual(kWeightLanes) || !qWeightLanes.SequenceEqual(vWeightLanes))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects q/k/v packed lanes to match, got q=[{string.Join(",", qWeightLanes)}], k=[{string.Join(",", kWeightLanes)}], v=[{string.Join(",", vWeightLanes)}].");
            }

            foreach (var (name, bias) in new[] { ("q", qBias), ("k", kBias), ("v", vBias) })
            {
                if (bias is not null && !GetVectorLanes(bias.ElemType).SequenceEqual(qWeightLanes))
                {
                    throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects {name} bias lanes [{string.Join(",", qWeightLanes)}], got [{string.Join(",", GetVectorLanes(bias.ElemType))}].");
                }
            }

            ValidatePackedProjectionShape("q", inputShape, qWeightShape, qOutputShape);
            ValidatePackedProjectionShape("k", inputShape, kWeightShape, kOutputShape);
            ValidatePackedProjectionShape("v", inputShape, vWeightShape, vOutputShape);
            ValidatePackedBiasShape("q", qBiasShape, qOutputShape);
            ValidatePackedBiasShape("k", kBiasShape, kOutputShape);
            ValidatePackedBiasShape("v", vBiasShape, vOutputShape);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear input/q output batch", inputShape[..^2], qOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear input/k output batch", inputShape[..^2], kOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear input/v output batch", inputShape[..^2], vOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear q weight/q output batch", qWeightShape[..^2], qOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear k weight/k output batch", kWeightShape[..^2], kOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear v weight/v output batch", vWeightShape[..^2], vOutputShape[..^2]);

            if (input.ElemType != GetScalarDataType(qWeight.ElemType) ||
                GetScalarDataType(qWeight.ElemType) != GetScalarDataType(kWeight.ElemType) ||
                GetScalarDataType(kWeight.ElemType) != GetScalarDataType(vWeight.ElemType))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects input and packed q/k/v weights to have the same scalar dtype, got input={input.ElemType}, q={qWeight.ElemType}, k={kWeight.ElemType}, v={vWeight.ElemType}.");
            }

            if (GetScalarDataType(qOutput.ElemType) != GetScalarDataType(kOutput.ElemType) ||
                GetScalarDataType(kOutput.ElemType) != GetScalarDataType(vOutput.ElemType))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects q/k/v outputs to have the same scalar dtype, got q={qOutput.ElemType}, k={kOutput.ElemType}, v={vOutput.ElemType}.");
            }

            var biasDType = qBias?.ElemType ?? kBias?.ElemType ?? vBias?.ElemType ?? qOutput.ElemType;
            var nPackedLaneCount = qWeightLanes[0];
            var nVectorLaneCount = qWeightLanes[1];
            _attrs["op"] = "packed_qkv_parallel_linear";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(qOutput.ElemType);
            _attrs["has_bias"] = qBias is not null || kBias is not null || vBias is not null;
            _attrs["q_shape"] = qOutputShape;
            _attrs["k_shape"] = kOutputShape;
            _attrs["v_shape"] = vOutputShape;
            _attrs["n_pack_lane"] = nPackedLaneCount;
            _attrs["n_lane"] = nVectorLaneCount;
            _attrs["n_scalar_lane"] = checked(nPackedLaneCount * nVectorLaneCount);
            _attrs["num_heads"] = qkv.NumHeads;
            _attrs["num_kv_heads"] = qkv.NumKvHeads;
            var qLogicalOutputShape = qOutputShape.ToArray();
            qLogicalOutputShape[^1] = MultiplyDim(qLogicalOutputShape[^1], checked(nPackedLaneCount * nVectorLaneCount));
            var useGemv = IsGemvMatmul(qLogicalOutputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "packed_qkv_gemv_compute" : "packed_qkv_matmul_compute");
            var templateModel = new PyNTTQKVParallelLinearTemplateModel(
                helperName,
                GetBufferScalarPointer(input),
                GetBufferScalarPointer(qWeight),
                GetBufferScalarPointer(kWeight),
                GetBufferScalarPointer(vWeight),
                qBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(qBias),
                kBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(kBias),
                vBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(vBias),
                GetBufferScalarPointer(qOutput),
                GetBufferScalarPointer(kOutput),
                GetBufferScalarPointer(vOutput),
                qBias is not null,
                kBias is not null,
                vBias is not null,
                GetPyNTTScalarDTypeName(input.ElemType),
                GetPyNTTScalarDTypeName(qWeight.ElemType),
                GetPyNTTScalarDTypeName(biasDType),
                GetPyNTTScalarDTypeName(qOutput.ElemType),
                GetScalarTritonDType(input.ElemType),
                GetScalarTritonDType(qWeight.ElemType),
                GetScalarTritonDType(biasDType),
                GetScalarTritonDType(qOutput.ElemType),
                inputShape,
                qWeightShape,
                kWeightShape,
                vWeightShape,
                qBiasShape,
                kBiasShape,
                vBiasShape,
                qOutputShape,
                kOutputShape,
                vOutputShape,
                GetBufferStrides(input),
                GetBufferStrides(qWeight),
                GetBufferStrides(kWeight),
                GetBufferStrides(vWeight),
                qBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(qBias),
                kBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(kBias),
                vBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(vBias),
                GetBufferStrides(qOutput),
                GetBufferStrides(kOutput),
                GetBufferStrides(vOutput),
                GetHierarchy(qOutput),
                $"{input.Name}, packed({qWeight.Name}, {kWeight.Name}, {vWeight.Name}) -> packed({qOutput.Name}, {kOutput.Name}, {vOutput.Name})")
            {
                PackedN = true,
                NPackedLaneCount = nPackedLaneCount,
                NVectorLaneCount = nVectorLaneCount,
            };

            if (_currentReductionState is { } reductionState)
            {
                PyNTTDimExpression[] GetLogicalOutputShape(PyNTTDimExpression[] shape)
                {
                    var logicalShape = shape.ToArray();
                    logicalShape[^1] = MultiplyDim(logicalShape[^1], checked(nPackedLaneCount * nVectorLaneCount));
                    return logicalShape;
                }

                EmitQKVParallelLinearReductionUpdate(
                    reductionState,
                    templateModel,
                    "triton/kernels/PackedQKVParallelLinear.py.jinja",
                    GetLogicalOutputShape(qOutputShape),
                    GetLogicalOutputShape(kOutputShape),
                    GetLogicalOutputShape(vOutputShape),
                    useGemv);
                return;
            }

            WriteHelperTemplate("triton/kernels/PackedQKVParallelLinear.py.jinja", templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private void EmitQKVParallelLinearReductionUpdate(
            ReductionState state,
            PyNTTQKVParallelLinearTemplateModel model,
            string templatePath,
            IReadOnlyList<PyNTTDimExpression> qLogicalOutputShape,
            IReadOnlyList<PyNTTDimExpression> kLogicalOutputShape,
            IReadOnlyList<PyNTTDimExpression> vLogicalOutputShape,
            bool useGemv)
        {
            var microKernel = RequireCurrentBlockMicroKernel("PyNTT QKVParallelLinear reduction");
            ApplyBlockMicroKernel(model, microKernel);
            var expectedKind = model.PackedN
                ? ReductionKernelKind.PackedQKVParallelLinear
                : ReductionKernelKind.QKVParallelLinear;
            if (state.Kind != expectedKind || state.Names.Length != 3)
            {
                throw new InvalidOperationException(
                    $"PyNTT reduction state {state.Kind} is incompatible with {expectedKind}.");
            }

            var logicalOutputShapes = new[]
            {
                qLogicalOutputShape,
                kLogicalOutputShape,
                vLogicalOutputShape,
            };
            if (logicalOutputShapes.Any(shape => shape.Count != 2))
            {
                throw new NotSupportedException(
                    $"PyNTT backend-private QKVParallelLinear accumulation currently requires rank-2 local tiles, got " +
                    string.Join(", ", logicalOutputShapes.Select(shape => $"[{ShapeText(shape)}]")) + ".");
            }

            if (!SameDim(qLogicalOutputShape[^2], kLogicalOutputShape[^2]) ||
                !SameDim(qLogicalOutputShape[^2], vLogicalOutputShape[^2]))
            {
                throw new NotSupportedException("PyNTT QKVParallelLinear reduction requires matching local M tile extents.");
            }

            var blockM = useGemv
                ? 1
                : GetReductionBlockExtent(
                    qLogicalOutputShape[^2],
                    "PyNTT QKVParallelLinear M",
                    16);
            var minimumBlockN = useGemv ? 32 : 16;
            string[] projectionNames = ["Q", "K", "V"];
            var blockNs = logicalOutputShapes
                .Select((shape, index) => GetReductionBlockExtent(
                    shape[^1],
                    $"PyNTT QKVParallelLinear {projectionNames[index]} N",
                    minimumBlockN))
                .ToArray();
            var blockK = GetMatmulReductionBlockKExtent(
                model.InputShape[^1],
                "PyNTT QKVParallelLinear K",
                useGemv);
            model.ReductionPhase = "accumulate";
            model.ReductionBlockM = blockM;
            model.ReductionBlockK = blockK;
            model.ReductionQBlockN = blockNs[0];
            model.ReductionKBlockN = blockNs[1];
            model.ReductionVBlockN = blockNs[2];
            var initializers = blockNs
                .Select(blockN => BuildMatrixReductionAccumulatorInitializer(
                    microKernel,
                    useGemv,
                    blockM,
                    blockN))
                .ToArray();
            ConfigureReductionState(
                state,
                initializers,
                templatePath,
                model.PackedN ? "packed_qkv_finalize" : "qkv_finalize",
                functionName => ApplyBlockMicroKernel(
                    new PyNTTQKVParallelLinearReductionFinalizeTemplateModel(
                    functionName,
                    model.QBias,
                    model.KBias,
                    model.VBias,
                    model.QOutput,
                    model.KOutput,
                    model.VOutput,
                    model.HasQBias,
                    model.HasKBias,
                    model.HasVBias,
                    model.BiasDType,
                    model.OutputDType,
                    model.BiasTritonDType,
                    model.OutputTritonDType,
                    model.QBiasShape,
                    model.KBiasShape,
                    model.VBiasShape,
                    model.QOutputShape,
                    model.KOutputShape,
                    model.VOutputShape,
                    model.QBiasStrides,
                    model.KBiasStrides,
                    model.VBiasStrides,
                    model.QOutputStrides,
                    model.KOutputStrides,
                    model.VOutputStrides,
                    model.PackedN,
                    model.NPackedLaneCount,
                    model.NVectorLaneCount,
                    blockM,
                    blockNs[0],
                    blockNs[1],
                    blockNs[2],
                    model.Comment), microKernel));

            WriteHelperTemplate(templatePath, model, requiresInline: true);
            var updateCall = BuildHelperCall(
                model.FunctionName,
                state.Names.Select(BuildRawPythonArgument).ToArray());
            WriteControlLine($"{string.Join(", ", state.Names)} = {updateCall}");
            state.UpdateCount++;
        }

        private void VisitMatMulGlu(Nncase.TIR.NTT.MatMulGlu matMulGlu, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 10 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer gateWeight ||
                args[2] is not TIR.Buffer upWeight ||
                args[9] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT MatMulGlu codegen expects input, gate/up weights, optional gate/up bias and scale values, and output TIR buffers.");
            }

            TIR.Buffer? GetOptionalBuffer(int index, string name) => args[index] switch
            {
                TIR.Buffer buffer => buffer,
                None => null,
                _ => throw new NotSupportedException($"PyNTT MatMulGlu expects {name} to be a TIR buffer or None, got {args[index].GetType().Name}."),
            };

            if (args.Skip(5).Take(4).Any(arg => arg is not None))
            {
                throw new NotSupportedException("PyNTT MatMulGlu codegen currently supports only None input/weight scales.");
            }

            var gateBias = GetOptionalBuffer(3, "gate bias");
            var upBias = GetOptionalBuffer(4, "up bias");
            SetComputeOp("matmul_glu");
            var inputShape = GetBufferActiveShape(input);
            var gateWeightShape = GetBufferActiveShape(gateWeight);
            var upWeightShape = GetBufferActiveShape(upWeight);
            var gateBiasShape = gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(gateBias);
            var upBiasShape = upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(upBias);
            var outputShape = GetBufferActiveShape(output);
            ValidateMinimumRank("PyNTT MatMulGlu input", inputShape, 2);
            ValidateMinimumRank("PyNTT MatMulGlu gate weight", gateWeightShape, 2);
            ValidateMinimumRank("PyNTT MatMulGlu up weight", upWeightShape, 2);
            ValidateMinimumRank("PyNTT MatMulGlu output", outputShape, 2);

            foreach (var (name, buffer) in new[]
            {
                ("input", input),
                ("gate weight", gateWeight),
                ("up weight", upWeight),
                ("output", output),
            })
            {
                var lanes = GetVectorLanes(buffer.ElemType);
                if (lanes.Length != 0)
                {
                    throw new NotSupportedException($"PyNTT MatMulGlu currently expects scalar {name} operands, got lanes [{string.Join(",", lanes)}].");
                }
            }

            ValidateMatMulGluProjectionShape("gate", inputShape, gateWeightShape, outputShape, packed: false);
            ValidateMatMulGluProjectionShape("up", inputShape, upWeightShape, outputShape, packed: false);
            ValidateMatMulGluBiasShape("gate", gateBiasShape, outputShape);
            ValidateMatMulGluBiasShape("up", upBiasShape, outputShape);
            ValidateBroadcastable("PyNTT MatMulGlu input/output batch", inputShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT MatMulGlu gate weight/output batch", gateWeightShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT MatMulGlu up weight/output batch", upWeightShape[..^2], outputShape[..^2]);

            if (input.ElemType != gateWeight.ElemType || gateWeight.ElemType != upWeight.ElemType)
            {
                throw new NotSupportedException($"PyNTT MatMulGlu expects input and gate/up weights to have the same dtype, got input={input.ElemType}, gate={gateWeight.ElemType}, up={upWeight.ElemType}.");
            }

            var biasDType = gateBias?.ElemType ?? upBias?.ElemType ?? output.ElemType;
            _attrs["op"] = "matmul_glu";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["has_bias"] = gateBias is not null || upBias is not null;
            _attrs["shape"] = outputShape;
            _attrs["glu_type"] = GetGluTypeName(matMulGlu.GluType);
            var useGemv = IsGemvMatmul(outputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "matmul_glu_gemv_compute" : "matmul_glu_matmul_compute");
            var templateModel = new PyNTTMatMulGluTemplateModel(
                helperName,
                GetBufferScalarPointer(input),
                GetBufferScalarPointer(gateWeight),
                GetBufferScalarPointer(upWeight),
                gateBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(gateBias),
                upBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(upBias),
                GetBufferScalarPointer(output),
                gateBias is not null,
                upBias is not null,
                GetGluTypeName(matMulGlu.GluType),
                GetPyNTTScalarDTypeName(input.ElemType),
                GetPyNTTScalarDTypeName(gateWeight.ElemType),
                GetPyNTTScalarDTypeName(biasDType),
                GetPyNTTScalarDTypeName(output.ElemType),
                GetScalarTritonDType(input.ElemType),
                GetScalarTritonDType(gateWeight.ElemType),
                GetScalarTritonDType(biasDType),
                GetScalarTritonDType(output.ElemType),
                inputShape,
                gateWeightShape,
                upWeightShape,
                gateBiasShape,
                upBiasShape,
                outputShape,
                GetBufferStrides(input),
                GetBufferStrides(gateWeight),
                GetBufferStrides(upWeight),
                gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(gateBias),
                upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(upBias),
                GetBufferStrides(output),
                $"{input.Name}, ({gateWeight.Name}, {upWeight.Name}) -> {output.Name}");

            if (_currentReductionState is { } reductionState)
            {
                EmitMatMulGluReductionUpdate(
                    reductionState,
                    templateModel,
                    "triton/kernels/MatMulGlu.py.jinja",
                    outputShape,
                    useGemv);
                return;
            }

            WriteHelperTemplate("triton/kernels/MatMulGlu.py.jinja", templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitPackedMatMulGlu(Nncase.TIR.NTT.PackedMatMulGlu matMulGlu, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 10 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer gateWeight ||
                args[2] is not TIR.Buffer upWeight ||
                args[9] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT PackedMatMulGlu codegen expects input, packed gate/up weights, optional packed gate/up bias and scale values, and packed output TIR buffers.");
            }

            TIR.Buffer? GetOptionalBuffer(int index, string name) => args[index] switch
            {
                TIR.Buffer buffer => buffer,
                None => null,
                _ => throw new NotSupportedException($"PyNTT PackedMatMulGlu expects {name} to be a TIR buffer or None, got {args[index].GetType().Name}."),
            };

            if (args.Skip(5).Take(4).Any(arg => arg is not None))
            {
                throw new NotSupportedException("PyNTT PackedMatMulGlu codegen currently supports only None input/weight scales.");
            }

            var gateBias = GetOptionalBuffer(3, "gate bias");
            var upBias = GetOptionalBuffer(4, "up bias");
            SetComputeOp("packed_matmul_glu");
            var inputShape = GetBufferActiveShape(input);
            var gateWeightShape = GetBufferActiveShape(gateWeight);
            var upWeightShape = GetBufferActiveShape(upWeight);
            var gateBiasShape = gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(gateBias);
            var upBiasShape = upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(upBias);
            var outputShape = GetBufferActiveShape(output);
            ValidateMinimumRank("PyNTT PackedMatMulGlu input", inputShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMulGlu gate weight", gateWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMulGlu up weight", upWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMulGlu output", outputShape, 2);

            var inputVectorLanes = GetVectorLanes(input.ElemType);
            if (inputVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"PyNTT PackedMatMulGlu expects scalar input operands, got lanes [{string.Join(",", inputVectorLanes)}].");
            }

            var gateWeightLanes = GetVectorLanes(gateWeight.ElemType);
            var upWeightLanes = GetVectorLanes(upWeight.ElemType);
            var outputLanes = GetVectorLanes(output.ElemType);
            ValidatePackedQKVLanes("gate", gateWeightLanes, outputLanes);
            ValidatePackedQKVLanes("up", upWeightLanes, outputLanes);
            if (!gateWeightLanes.SequenceEqual(upWeightLanes))
            {
                throw new NotSupportedException($"PyNTT PackedMatMulGlu expects gate/up packed lanes to match, got gate=[{string.Join(",", gateWeightLanes)}], up=[{string.Join(",", upWeightLanes)}].");
            }

            foreach (var (name, bias) in new[] { ("gate", gateBias), ("up", upBias) })
            {
                if (bias is not null && !GetVectorLanes(bias.ElemType).SequenceEqual(gateWeightLanes))
                {
                    throw new NotSupportedException($"PyNTT PackedMatMulGlu expects {name} bias lanes [{string.Join(",", gateWeightLanes)}], got [{string.Join(",", GetVectorLanes(bias.ElemType))}].");
                }
            }

            ValidateMatMulGluProjectionShape("gate", inputShape, gateWeightShape, outputShape, packed: true);
            ValidateMatMulGluProjectionShape("up", inputShape, upWeightShape, outputShape, packed: true);
            ValidateMatMulGluBiasShape("gate", gateBiasShape, outputShape);
            ValidateMatMulGluBiasShape("up", upBiasShape, outputShape);
            ValidateBroadcastable("PyNTT PackedMatMulGlu input/output batch", inputShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedMatMulGlu gate weight/output batch", gateWeightShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedMatMulGlu up weight/output batch", upWeightShape[..^2], outputShape[..^2]);

            if (input.ElemType != GetScalarDataType(gateWeight.ElemType) ||
                GetScalarDataType(gateWeight.ElemType) != GetScalarDataType(upWeight.ElemType))
            {
                throw new NotSupportedException($"PyNTT PackedMatMulGlu expects input and packed gate/up weights to have the same scalar dtype, got input={input.ElemType}, gate={gateWeight.ElemType}, up={upWeight.ElemType}.");
            }

            var biasDType = gateBias?.ElemType ?? upBias?.ElemType ?? output.ElemType;
            var nPackedLaneCount = gateWeightLanes[0];
            var nVectorLaneCount = gateWeightLanes[1];
            _attrs["op"] = "packed_matmul_glu";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["has_bias"] = gateBias is not null || upBias is not null;
            _attrs["shape"] = outputShape;
            _attrs["n_pack_lane"] = nPackedLaneCount;
            _attrs["n_lane"] = nVectorLaneCount;
            _attrs["n_scalar_lane"] = checked(nPackedLaneCount * nVectorLaneCount);
            _attrs["glu_type"] = GetGluTypeName(matMulGlu.GluType);
            var logicalOutputShape = outputShape.ToArray();
            logicalOutputShape[^1] = MultiplyDim(logicalOutputShape[^1], checked(nPackedLaneCount * nVectorLaneCount));
            var useGemv = IsGemvMatmul(logicalOutputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "packed_matmul_glu_gemv_compute" : "packed_matmul_glu_matmul_compute");
            var templateModel = new PyNTTMatMulGluTemplateModel(
                helperName,
                GetBufferScalarPointer(input),
                GetBufferScalarPointer(gateWeight),
                GetBufferScalarPointer(upWeight),
                gateBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(gateBias),
                upBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(upBias),
                GetBufferScalarPointer(output),
                gateBias is not null,
                upBias is not null,
                GetGluTypeName(matMulGlu.GluType),
                GetPyNTTScalarDTypeName(input.ElemType),
                GetPyNTTScalarDTypeName(gateWeight.ElemType),
                GetPyNTTScalarDTypeName(biasDType),
                GetPyNTTScalarDTypeName(output.ElemType),
                GetScalarTritonDType(input.ElemType),
                GetScalarTritonDType(gateWeight.ElemType),
                GetScalarTritonDType(biasDType),
                GetScalarTritonDType(output.ElemType),
                inputShape,
                gateWeightShape,
                upWeightShape,
                gateBiasShape,
                upBiasShape,
                outputShape,
                GetBufferStrides(input),
                GetBufferStrides(gateWeight),
                GetBufferStrides(upWeight),
                gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(gateBias),
                upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(upBias),
                GetBufferStrides(output),
                $"{input.Name}, packed({gateWeight.Name}, {upWeight.Name}) -> packed({output.Name})")
            {
                PackedN = true,
                NPackedLaneCount = nPackedLaneCount,
                NVectorLaneCount = nVectorLaneCount,
            };

            if (_currentReductionState is { } reductionState)
            {
                EmitMatMulGluReductionUpdate(
                    reductionState,
                    templateModel,
                    "triton/kernels/PackedMatMulGlu.py.jinja",
                    logicalOutputShape,
                    useGemv);
                return;
            }

            WriteHelperTemplate("triton/kernels/PackedMatMulGlu.py.jinja", templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private void EmitMatMulGluReductionUpdate(
            ReductionState state,
            PyNTTMatMulGluTemplateModel model,
            string templatePath,
            IReadOnlyList<PyNTTDimExpression> logicalOutputShape,
            bool useGemv)
        {
            var microKernel = RequireCurrentBlockMicroKernel("PyNTT MatMulGlu reduction");
            ApplyBlockMicroKernel(model, microKernel);
            var expectedKind = model.PackedN
                ? ReductionKernelKind.PackedMatMulGlu
                : ReductionKernelKind.MatMulGlu;
            if (state.Kind != expectedKind || state.Names.Length != 2)
            {
                throw new InvalidOperationException(
                    $"PyNTT reduction state {state.Kind} is incompatible with {expectedKind}.");
            }

            if (logicalOutputShape.Count != 2)
            {
                throw new NotSupportedException(
                    $"PyNTT backend-private MatMulGlu accumulation currently requires a rank-2 local tile, got [{ShapeText(logicalOutputShape)}].");
            }

            var blockM = useGemv
                ? 1
                : GetReductionBlockExtent(
                    logicalOutputShape[^2],
                    "PyNTT MatMulGlu M",
                    16);
            var minimumBlockN = useGemv ? 32 : 16;
            var blockN = GetReductionBlockExtent(
                logicalOutputShape[^1],
                "PyNTT MatMulGlu N",
                minimumBlockN);
            var blockK = GetMatmulReductionBlockKExtent(
                model.InputShape[^1],
                "PyNTT MatMulGlu K",
                useGemv);
            model.ReductionPhase = "accumulate";
            model.ReductionBlockM = blockM;
            model.ReductionBlockN = blockN;
            model.ReductionBlockK = blockK;
            ReductionAccumulatorInitializer[] initializers =
            [
                BuildMatrixReductionAccumulatorInitializer(microKernel, useGemv, blockM, blockN),
                BuildMatrixReductionAccumulatorInitializer(microKernel, useGemv, blockM, blockN),
            ];
            ConfigureReductionState(
                state,
                initializers,
                templatePath,
                model.PackedN ? "packed_matmul_glu_finalize" : "matmul_glu_finalize",
                functionName => ApplyBlockMicroKernel(
                    new PyNTTMatMulGluReductionFinalizeTemplateModel(
                    functionName,
                    model.GateBias,
                    model.UpBias,
                    model.Output,
                    model.HasGateBias,
                    model.HasUpBias,
                    model.GluType,
                    model.BiasDType,
                    model.OutputDType,
                    model.BiasTritonDType,
                    model.OutputTritonDType,
                    model.GateBiasShape,
                    model.UpBiasShape,
                    model.OutputShape,
                    model.GateBiasStrides,
                    model.UpBiasStrides,
                    model.OutputStrides,
                    model.PackedN,
                    model.NPackedLaneCount,
                    model.NVectorLaneCount,
                    blockM,
                    blockN,
                    model.Comment), microKernel));

            WriteHelperTemplate(templatePath, model, requiresInline: true);
            var updateCall = BuildHelperCall(
                model.FunctionName,
                state.Names.Select(BuildRawPythonArgument).ToArray());
            WriteControlLine($"{string.Join(", ", state.Names)} = {updateCall}");
            state.UpdateCount++;
        }

        private static void ValidateMatMulGluProjectionShape(
            string name,
            PyNTTDimExpression[] inputShape,
            PyNTTDimExpression[] weightShape,
            PyNTTDimExpression[] outputShape,
            bool packed)
        {
            var inputK = inputShape[^1];
            var inputM = inputShape[^2];
            var weightK = packed ? weightShape[^1] : weightShape[^2];
            var weightN = packed ? weightShape[^2] : weightShape[^1];
            var outputM = outputShape[^2];
            var outputN = outputShape[^1];
            if (!SameDim(inputK, weightK) ||
                !SameDim(outputM, inputM) ||
                !SameDim(outputN, weightN))
            {
                var weightLayout = packed ? "weight=[...,N,K]<Nr,lane>" : "weight=[...,K,N]";
                var outputLayout = packed ? "output=[...,M,N]<Nr,lane>" : "output=[...,M,N]";
                throw new NotSupportedException($"PyNTT MatMulGlu {name} projection expects compatible local tile shapes input=[...,M,K], {weightLayout}, {outputLayout}, got input=[{ShapeText(inputShape)}], weight=[{ShapeText(weightShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private static void ValidateMatMulGluBiasShape(
            string name,
            PyNTTDimExpression[] biasShape,
            PyNTTDimExpression[] outputShape)
        {
            if (biasShape.Length == 0)
            {
                return;
            }

            ValidateRank($"PyNTT MatMulGlu {name} bias", biasShape, 1);
            if (!SameDim(biasShape[^1], outputShape[^1]))
            {
                throw new NotSupportedException($"PyNTT MatMulGlu {name} bias last dimension should match the local output N tile, got bias=[{ShapeText(biasShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private static void ValidateBiasShape(string name, PyNTTDimExpression[] biasShape, PyNTTDimExpression[] outputShape)
        {
            if (biasShape.Length == 0)
            {
                return;
            }

            ValidateRank($"PyNTT QKVParallelLinear {name} bias", biasShape, 1);
            if (!SameDim(biasShape[^1], outputShape[^1]))
            {
                throw new NotSupportedException($"PyNTT QKVParallelLinear {name} bias last dimension should match output N, got bias=[{ShapeText(biasShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private static void ValidateProjectionShape(string name, PyNTTDimExpression[] inputShape, PyNTTDimExpression[] weightShape, PyNTTDimExpression[] outputShape)
        {
            var inputK = inputShape[^1];
            var inputM = inputShape[^2];
            var weightK = weightShape[^2];
            var weightN = weightShape[^1];
            if (!SameDim(inputK, weightK) ||
                !SameDim(outputShape[^2], inputM) ||
                !SameDim(outputShape[^1], weightN))
            {
                throw new NotSupportedException($"PyNTT QKVParallelLinear {name} projection expects input=[...,M,K], weight=[...,K,N], output=[...,M,N], got input=[{ShapeText(inputShape)}], weight=[{ShapeText(weightShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private static void ValidatePackedQKVLanes(string name, IReadOnlyList<int> weightLanes, IReadOnlyList<int> outputLanes)
        {
            if (weightLanes.Count != 2)
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects {name} weight lanes [Nr,lane], got [{string.Join(",", weightLanes)}].");
            }

            if (!weightLanes.SequenceEqual(outputLanes))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects {name} output lanes [{string.Join(",", weightLanes)}], got [{string.Join(",", outputLanes)}].");
            }
        }

        private static void ValidatePackedBiasShape(string name, PyNTTDimExpression[] biasShape, PyNTTDimExpression[] outputShape)
        {
            if (biasShape.Length == 0)
            {
                return;
            }

            ValidateRank($"PyNTT PackedQKVParallelLinear {name} bias", biasShape, 1);
            if (!SameDim(biasShape[^1], outputShape[^1]))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear {name} bias last dimension should match packed output N, got bias=[{ShapeText(biasShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private static void ValidatePackedProjectionShape(string name, PyNTTDimExpression[] inputShape, PyNTTDimExpression[] weightShape, PyNTTDimExpression[] outputShape)
        {
            var inputK = inputShape[^1];
            var inputM = inputShape[^2];
            var weightN = weightShape[^2];
            var weightK = weightShape[^1];
            if (!SameDim(inputK, weightK) ||
                !SameDim(outputShape[^2], inputM) ||
                !SameDim(outputShape[^1], weightN))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear {name} projection expects input=[...,M,K], weight=[...,N,K]<Nr,lane>, output=[...,M,N]<Nr,lane>, got input=[{ShapeText(inputShape)}], weight=[{ShapeText(weightShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private void VisitSUMMA(Nncase.TIR.NTT.SUMMA summa, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 5 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT SUMMA codegen expects lhs, rhs, and output TIR buffers.");
            }

            if (summa.TransposeA || summa.TransposeB)
            {
                throw new NotSupportedException("PyNTT SUMMA codegen does not support transposed inputs yet.");
            }

            if (GetScalarBool(args[3], "PyNTT SUMMA loadC"))
            {
                throw new NotSupportedException("PyNTT SUMMA codegen does not support loadC yet.");
            }

            SetComputeOp("matmul");
            var lhsShape = GetBufferShape(lhs);
            var rhsShape = GetBufferShape(rhs);
            var outputShape = GetBufferShape(output);
            var lhsGlobalShape = GetBufferGlobalShape(lhs);
            var rhsGlobalShape = GetBufferGlobalShape(rhs);
            var outputGlobalShape = GetBufferGlobalShape(output);
            ValidateMinimumRank("PyNTT SUMMA lhs", lhsShape, 2);
            ValidateMinimumRank("PyNTT SUMMA rhs", rhsShape, 2);
            ValidateMinimumRank("PyNTT SUMMA output", outputShape, 2);
            if (lhsShape.Length != 2 || rhsShape.Length != 2 || outputShape.Length != 2)
            {
                throw new NotSupportedException($"PyNTT SUMMA currently supports rank-2 matmul only, got lhs rank {lhsShape.Length}, rhs rank {rhsShape.Length}, output rank {outputShape.Length}.");
            }

            var dimInfo = Nncase.IR.NTT.VectorizedMatMul.GetDimInfo(false, false, lhsShape.Length, rhsShape.Length);
            var lhsVectorLanes = GetVectorLanes(lhs.ElemType);
            var rhsVectorLanes = GetVectorLanes(rhs.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            var rhsNVectorLaneCount = 1;
            var outputNVectorLaneCount = 1;
            if (!summa.LhsVectorizedAxes.IsDefaultOrEmpty)
            {
                throw new NotSupportedException($"PyNTT SUMMA currently supports only RHS N-axis vectorization, got lhs axes [{string.Join(",", summa.LhsVectorizedAxes)}].");
            }

            if (!summa.RhsVectorizedAxes.IsDefaultOrEmpty)
            {
                if (summa.RhsVectorizedAxes.Count != 1 || summa.RhsVectorizedAxes[0] != dimInfo.Rn || rhsVectorLanes.Length != 1)
                {
                    throw new NotSupportedException($"PyNTT SUMMA currently supports only one RHS N-axis vector lane, got rhs axes [{string.Join(",", summa.RhsVectorizedAxes)}] and lanes [{string.Join(",", rhsVectorLanes)}].");
                }

                rhsNVectorLaneCount = rhsVectorLanes[0];
            }

            if (lhsVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"PyNTT SUMMA currently supports scalar lhs operands only, got lhs lanes [{string.Join(",", lhsVectorLanes)}].");
            }

            if (rhsNVectorLaneCount == 1)
            {
                if (outputVectorLanes.Length != 0)
                {
                    throw new NotSupportedException($"PyNTT SUMMA scalar RHS requires a scalar output, got output lanes [{string.Join(",", outputVectorLanes)}].");
                }
            }
            else if (outputVectorLanes.Length != 1 || outputVectorLanes[0] != rhsNVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT SUMMA vectorized RHS requires the same output N lane so both buffers have direct physical/lane coordinates, got output lanes [{string.Join(",", outputVectorLanes)}] and rhs N lane {rhsNVectorLaneCount}.");
            }
            else
            {
                outputNVectorLaneCount = outputVectorLanes[0];
            }

            var outputGlobalNCompatible = SameDim(outputGlobalShape[^1], rhsGlobalShape[^1]);
            if (!SameDim(lhsGlobalShape[^1], rhsGlobalShape[^2]) ||
                !SameDim(outputGlobalShape[^2], lhsGlobalShape[^2]) ||
                !outputGlobalNCompatible)
            {
                throw new NotSupportedException($"PyNTT SUMMA expects compatible global matrix shapes, got lhs=[{ShapeText(lhsGlobalShape)}], rhs=[{ShapeText(rhsGlobalShape)}], output=[{ShapeText(outputGlobalShape)}].");
            }

            var lhsRef = ResolveByteAddressedBufferRef(lhs);
            var rhsRef = ResolveByteAddressedBufferRef(rhs);
            var outputRef = ResolveByteAddressedBufferRef(output);
            var scale = GetScalarFloat(args[4], "PyNTT SUMMA scale", 1.0f);
            _attrs["op"] = "matmul";
            _attrs["summa"] = true;
            _attrs["requires_grid_barrier"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["scale"] = scale;
            var helperName = GetNextHelperName("summa_compute");
            WriteHelperTemplate(
                "triton/kernels/Summa.py.jinja",
                new PyNTTSummaTemplateModel(
                    helperName,
                    lhsRef.BaseName,
                    lhsRef.OffsetBytes,
                    lhsRef.PoolStrideBytes,
                    lhsRef.AddressSpace,
                    rhsRef.BaseName,
                    rhsRef.OffsetBytes,
                    rhsRef.PoolStrideBytes,
                    rhsRef.AddressSpace,
                    outputRef.BaseName,
                    outputRef.OffsetBytes,
                    outputRef.PoolStrideBytes,
                    outputRef.AddressSpace,
                    GetPyNTTScalarDTypeName(lhs.ElemType),
                    GetPyNTTScalarDTypeName(rhs.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(lhs.ElemType),
                    GetScalarTritonDType(rhs.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    outputShape,
                    lhsGlobalShape,
                    rhsGlobalShape,
                    outputGlobalShape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
                    GetBufferSplitAxes(lhs, lhsShape.Length),
                    GetBufferSplitAxes(rhs, rhsShape.Length),
                    GetBufferSplitAxes(output, outputShape.Length),
                    GetHierarchy(output),
                    rhsNVectorLaneCount,
                    outputNVectorLaneCount,
                    rhsVectorLanes,
                    outputVectorLanes,
                    scale.ToString("R", CultureInfo.InvariantCulture),
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
            WriteBarrier(HelperBarrierKind.Grid);
            WriteLine(BuildHelperCall(helperName), HelperBarrierKind.Grid);
        }

        private void VisitMatmulLike(
            string context,
            IRArray<int> lhsVectorizedAxes,
            IRArray<int> rhsVectorizedAxes,
            bool transposeA,
            bool transposeB,
            IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 5 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException($"{context} codegen expects lhs, rhs, and output TIR buffers.");
            }

            var loadCExpression = GetScalarBoolExpression(args[3], $"{context} loadC");

            SetComputeOp("matmul");
            var lhsShape = GetBufferActiveShape(lhs);
            var rhsShape = GetBufferActiveShape(rhs);
            var outputShape = GetBufferActiveShape(output);
            ValidateMinimumRank($"{context} lhs", lhsShape, 2);
            ValidateMinimumRank($"{context} rhs", rhsShape, 2);
            ValidateMinimumRank($"{context} output", outputShape, 2);
            var dimInfo = Nncase.IR.NTT.VectorizedMatMul.GetDimInfo(transposeA, transposeB, lhsShape.Length, rhsShape.Length);
            var lhsVectorLanes = GetVectorLanes(lhs.ElemType);
            var rhsVectorLanes = GetVectorLanes(rhs.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            var rhsNVectorLaneCount = 1;
            var outputNVectorLaneCount = 1;
            if (!lhsVectorizedAxes.IsDefaultOrEmpty)
            {
                throw new NotSupportedException($"{context} currently supports only RHS N-axis vectorization, got lhs axes [{string.Join(",", lhsVectorizedAxes)}].");
            }

            if (!rhsVectorizedAxes.IsDefaultOrEmpty)
            {
                if (rhsVectorizedAxes.Count != 1 || rhsVectorizedAxes[0] != dimInfo.Rn || rhsVectorLanes.Length != 1)
                {
                    throw new NotSupportedException($"{context} currently supports only one RHS N-axis vector lane, got rhs axes [{string.Join(",", rhsVectorizedAxes)}] and lanes [{string.Join(",", rhsVectorLanes)}].");
                }

                rhsNVectorLaneCount = rhsVectorLanes[0];
            }

            if (lhsVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"{context} currently supports scalar lhs operands only, got lhs lanes [{string.Join(",", lhsVectorLanes)}].");
            }

            if (outputVectorLanes.Length != 0)
            {
                if (outputVectorLanes.Length != 1 || outputVectorLanes[0] != rhsNVectorLaneCount || rhsNVectorLaneCount == 1)
                {
                    throw new NotSupportedException($"{context} currently supports only RHS N-axis vectorization producing the same output N lane, got output lanes [{string.Join(",", outputVectorLanes)}] and rhs N lane {rhsNVectorLaneCount}.");
                }

                outputNVectorLaneCount = outputVectorLanes[0];
            }

            var lhsK = transposeA ? lhsShape[^2] : lhsShape[^1];
            var rhsK = transposeB ? rhsShape[^1] : rhsShape[^2];
            var lhsM = transposeA ? lhsShape[^1] : lhsShape[^2];
            var rhsN = transposeB ? rhsShape[^2] : rhsShape[^1];
            var rhsFullN = MultiplyDim(rhsN, rhsNVectorLaneCount);
            var outputNCompatible = outputNVectorLaneCount == 1
                ? rhsNVectorLaneCount == 1 ? SameDim(outputShape[^1], rhsN) : CanFitPaddedDim(outputShape[^1], rhsFullN)
                : SameDim(outputShape[^1], rhsN);
            if (!SameDim(lhsK, rhsK) || !SameDim(outputShape[^2], lhsM) || !outputNCompatible)
            {
                throw new NotSupportedException($"{context} expects compatible matrix shapes, got lhs=[{ShapeText(lhsShape)}], rhs=[{ShapeText(rhsShape)}], output=[{ShapeText(outputShape)}].");
            }

            ValidateBroadcastable($"{context} lhs batch", lhsShape[..^2], outputShape[..^2]);
            ValidateBroadcastable($"{context} rhs batch", rhsShape[..^2], outputShape[..^2]);
            var scale = GetScalarFloat(args[4], $"{context} scale", 1.0f);
            _attrs["op"] = "matmul";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["transpose_a"] = transposeA;
            _attrs["transpose_b"] = transposeB;
            _attrs["scale"] = scale;
            var useGemv = IsGemvMatmul(outputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "gemv_compute" : "matmul_compute");
            var templateModel = new PyNTTMatmulTemplateModel(
                helperName,
                GetBufferScalarPointer(lhs),
                GetBufferScalarPointer(rhs),
                GetBufferScalarPointer(output),
                GetPyNTTScalarDTypeName(lhs.ElemType),
                GetPyNTTScalarDTypeName(rhs.ElemType),
                GetPyNTTScalarDTypeName(output.ElemType),
                GetScalarTritonDType(lhs.ElemType),
                GetScalarTritonDType(rhs.ElemType),
                GetScalarTritonDType(output.ElemType),
                lhsShape,
                rhsShape,
                outputShape,
                GetBufferStrides(lhs),
                GetBufferStrides(rhs),
                GetBufferStrides(output),
                transposeA,
                transposeB,
                GetHierarchy(output),
                rhsNVectorLaneCount,
                outputNVectorLaneCount,
                scale.ToString("R", CultureInfo.InvariantCulture),
                $"{lhs.Name}, {rhs.Name} -> {output.Name}")
            {
                LoadCExpression = loadCExpression,
            };

            if (_currentReductionState is { } reductionState)
            {
                var logicalOutputShape = outputShape.ToArray();
                logicalOutputShape[^1] = MultiplyDim(logicalOutputShape[^1], outputNVectorLaneCount);
                EmitMatmulReductionUpdate(
                    reductionState,
                    ReductionKernelKind.Matmul,
                    templateModel,
                    useGemv ? "triton/kernels/Gemv.py.jinja" : "triton/kernels/Matmul.py.jinja",
                    logicalOutputShape,
                    useGemv);
                return;
            }

            WriteHelperTemplate(
                useGemv ? "triton/kernels/Gemv.py.jinja" : "triton/kernels/Matmul.py.jinja",
                templateModel);
            WriteHelperInvocation(helperName);
        }

        private void EmitMatmulReductionUpdate(
            ReductionState state,
            ReductionKernelKind expectedKind,
            PyNTTMatmulTemplateModel model,
            string templatePath,
            IReadOnlyList<PyNTTDimExpression> logicalOutputShape,
            bool useGemv)
        {
            var microKernel = RequireCurrentBlockMicroKernel("PyNTT Matmul reduction");
            ApplyBlockMicroKernel(model, microKernel);
            if (state.Kind != expectedKind || state.Names.Length != 1)
            {
                throw new InvalidOperationException(
                    $"PyNTT reduction state {state.Kind} is incompatible with {expectedKind}.");
            }

            if (logicalOutputShape.Count != 2)
            {
                throw new NotSupportedException(
                    $"PyNTT backend-private Matmul accumulation currently requires a rank-2 local tile, got [{ShapeText(logicalOutputShape)}].");
            }

            var blockM = useGemv
                ? 1
                : GetReductionBlockExtent(
                    logicalOutputShape[^2],
                    "PyNTT Matmul M",
                    16);
            var minimumBlockN = useGemv ? 32 : 16;
            var blockN = GetReductionBlockExtent(
                logicalOutputShape[^1],
                "PyNTT Matmul N",
                minimumBlockN);
            var lhsK = model.TransposeA ? model.LhsShape[^2] : model.LhsShape[^1];
            var blockK = GetMatmulReductionBlockKExtent(
                lhsK,
                "PyNTT Matmul K",
                useGemv);
            model.ReductionPhase = "accumulate";
            model.ReductionBlockM = blockM;
            model.ReductionBlockN = blockN;
            model.ReductionBlockK = blockK;
            var accumulatorNGroupWidth = expectedKind == ReductionKernelKind.PackedMatmul
                ? checked(model.OutputNPackedLaneCount * model.OutputNVectorLaneCount)
                : 1;
            var initializer = BuildMatrixReductionAccumulatorInitializer(
                microKernel,
                useGemv,
                blockM,
                blockN,
                accumulatorNGroupWidth);
            ConfigureReductionState(
                state,
                [initializer],
                templatePath,
                expectedKind == ReductionKernelKind.PackedMatmul ? "packed_matmul_finalize" : "matmul_finalize",
                functionName => ApplyBlockMicroKernel(
                    new PyNTTMatmulReductionFinalizeTemplateModel(
                    functionName,
                    model.Output,
                    model.OutputDType,
                    model.OutputTritonDType,
                    model.OutputShape,
                    model.OutputStrides,
                    model.OutputNPackedLaneCount,
                    model.OutputNVectorLaneCount,
                    model.Scale,
                    useGemv,
                    blockM,
                    blockN,
                    model.Comment), microKernel));

            WriteHelperTemplate(templatePath, model, requiresInline: true);
            var updateCall = BuildHelperCall(model.FunctionName, BuildRawPythonArgument(state.Names[0]));
            WriteControlLine($"{state.Names[0]} = {updateCall}");

            state.UpdateCount++;
        }

        private static bool IsGemvMatmul(IReadOnlyList<PyNTTDimExpression> outputShape)
            => outputShape.Count >= 2 && outputShape[^2].MaxValue == 1;

        private void VisitReduce(Nncase.TIR.NTT.Reduce reduce, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Reduce codegen expects input and output TIR buffers.");
            }

            EnsureEmpty("PyNTT Reduce vectorized axes", reduce.VectorizedAxes);
            SetComputeOp("reduce");
            var inputShape = GetBufferActiveShape(input);
            var outputShape = GetBufferActiveShape(output);
            var axes = NormalizeAxes(reduce.Axes.ToArray(), inputShape.Length, "PyNTT Reduce");
            ValidateReduceShape("PyNTT Reduce", inputShape, outputShape, axes, reduce.KeepDims);
            _attrs["op"] = GetReduceOpName(reduce.ReduceOp);
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axes"] = axes;
            if (axes.Length == 1)
            {
                _attrs["axis"] = axes[0];
            }

            _attrs["keep_dims"] = reduce.KeepDims;
            var helperName = GetNextHelperName("reduce_compute");
            var reduceOp = (string)_attrs["op"];
            var reduceElementCount = Product(axes.Select(axis => inputShape[axis]));
            var templateModel = new PyNTTReduceTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    axes,
                    reduce.KeepDims,
                    reduceOp,
                    GetReduceInitValue(reduce.ReduceOp),
                    GetReduceUpdateExpression(reduce.ReduceOp),
                    GetReduceFinalizeExpression(reduce.ReduceOp, reduceElementCount),
                    $"{input.Name} -> {output.Name}");

            if (_currentReductionState is { } reductionState)
            {
                EmitReduceReductionUpdate(reductionState, reduce, input, output, templateModel);
                return;
            }

            if (GetScalarBool(args[2], "reduce loadPrevious"))
            {
                throw new NotSupportedException("PyNTT Reduce codegen does not support loadPrevious yet.");
            }

            WriteHelperTemplate("triton/kernels/Reduce.py.jinja", templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private void EmitReduceReductionUpdate(
            ReductionState state,
            Nncase.TIR.NTT.Reduce reduce,
            TIR.Buffer input,
            TIR.Buffer output,
            PyNTTReduceTemplateModel model)
        {
            var microKernel = RequireCurrentBlockMicroKernel("PyNTT Reduce reduction");
            ApplyBlockMicroKernel(model, microKernel);
            var trackElementCount = reduce.ReduceOp == ReduceOp.Mean;
            var expectedStateCount = trackElementCount ? 2 : 1;
            if (state.Kind != ReductionKernelKind.Reduce || state.Names.Length != expectedStateCount)
            {
                throw new InvalidOperationException(
                    $"PyNTT reduction state {state.Kind} with {state.Names.Length} physical states is incompatible with {reduce.ReduceOp} Reduce.");
            }

            if (GetVectorLaneElementCount(input.ElemType) != 1 || GetVectorLaneElementCount(output.ElemType) != 1)
            {
                throw new NotSupportedException(
                    $"PyNTT backend-private Reduce currently requires scalar element types, got input={input.ElemType}, output={output.ElemType}.");
            }

            var outputElements = Product(model.OutputShape);
            var blockSize = GetReductionBlockExtent(outputElements, "PyNTT Reduce output tile");
            var scalarInputType = GetScalarDataType(input.ElemType);
            var accumulatorType = DataTypes.IsFloat(scalarInputType)
                ? "tl.float32"
                : GetScalarTritonDType(scalarInputType);
            var initValue = GetReduceInitValue(reduce.ReduceOp, scalarInputType);
            model = model with
            {
                InitValue = initValue,
                FinalizeExpression = trackElementCount
                    ? "acc / reduced_element_count.to(tl.float32)"
                    : model.FinalizeExpression,
            };
            model.ReductionPhase = "accumulate";
            model.ReductionBlockSize = blockSize;
            model.AccumulatorTritonDType = accumulatorType;
            model.TrackReductionElementCount = trackElementCount;
            ReductionAccumulatorInitializer[] initializers = trackElementCount
                ? [
                    ReductionAccumulatorInitializer.Register($"tl.full(({blockSize},), {initValue}, {accumulatorType})"),
                    ReductionAccumulatorInitializer.Register("tl.full((), 0, tl.int64)"),
                ]
                : [ReductionAccumulatorInitializer.Register($"tl.full(({blockSize},), {initValue}, {accumulatorType})")];
            ConfigureReductionState(
                state,
                initializers,
                "triton/kernels/Reduce.py.jinja",
                "reduce_finalize",
                functionName => ApplyBlockMicroKernel(
                    new PyNTTReduceReductionFinalizeTemplateModel(
                    functionName,
                    model.Output,
                    model.OutputDType,
                    model.OutputTritonDType,
                    model.OutputShape,
                    model.OutputStrides,
                    model.FinalizeExpression,
                    blockSize,
                    trackElementCount,
                    model.Comment), microKernel));

            WriteHelperTemplate("triton/kernels/Reduce.py.jinja", model, requiresInline: true);
            var updateCall = BuildHelperCall(
                model.FunctionName,
                state.Names.Select(BuildRawPythonArgument).ToArray());
            WriteControlLine($"{string.Join(", ", state.Names)} = {updateCall}");
            state.UpdateCount++;
        }

        private void VisitRoPE(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer cos ||
                args[2] is not TIR.Buffer sin ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT RoPE codegen expects input, cos, sin, and output TIR buffers.");
            }

            SetComputeOp("rope");
            var inputShape = GetBufferShape(input);
            var cosShape = GetBufferShape(cos);
            var sinShape = GetBufferShape(sin);
            var outputShape = GetBufferShape(output);
            var inputVectorLanes = GetVectorLanes(input.ElemType);
            var cosVectorLanes = GetVectorLanes(cos.ElemType);
            var sinVectorLanes = GetVectorLanes(sin.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var cosVectorLaneCount = GetVectorLaneElementCount(cos.ElemType);
            var sinVectorLaneCount = GetVectorLaneElementCount(sin.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);
            var sinCosVectorPackFactor = GetRoPESinCosVectorPackFactor(inputVectorLanes, cosVectorLanes, sinVectorLanes, outputVectorLanes);

            var rotaryAxis = outputShape.Length - 1;
            if (rotaryAxis < 0)
            {
                throw new NotSupportedException($"PyNTT RoPE requires a non-scalar output, got [{ShapeText(outputShape)}].");
            }

            ValidateRoPEShape("PyNTT RoPE", inputShape, cosShape, sinShape, outputShape, rotaryAxis, outputVectorLaneCount, sinCosVectorPackFactor);
            if (GetShardAxis(output) == rotaryAxis || GetShardAxis(input) == rotaryAxis)
            {
                throw new NotSupportedException("PyNTT RoPE codegen does not support sharding along the rotary dimension yet.");
            }

            _attrs["op"] = "rope";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            var helperName = GetNextHelperName("rope_compute");
            WriteHelperTemplate(
                "triton/kernels/RoPE.py.jinja",
                new PyNTTRoPETemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(cos),
                    GetBufferScalarPointer(sin),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(cos.ElemType),
                    GetPyNTTScalarDTypeName(sin.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(cos.ElemType),
                    GetScalarTritonDType(sin.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    cosShape,
                    sinShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(cos),
                    GetBufferStrides(sin),
                    GetBufferStrides(output),
                    inputVectorLanes,
                    cosVectorLanes,
                    sinVectorLanes,
                    outputVectorLanes,
                    inputVectorLaneCount,
                    cosVectorLaneCount,
                    sinVectorLaneCount,
                    outputVectorLaneCount,
                    sinCosVectorPackFactor,
                    rotaryAxis,
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void VisitLayerNorm(Nncase.TIR.NTT.VectorizedLayerNorm layerNorm, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 5 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer scale ||
                args[2] is not TIR.Buffer bias ||
                args[4] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT LayerNorm codegen expects input, scale, bias, postScale, and output TIR buffers.");
            }

            if (args[3] is not None)
            {
                throw new NotSupportedException("PyNTT LayerNorm codegen does not support postScale yet.");
            }

            SetComputeOp("layer_norm");
            var inputShape = GetBufferShape(input);
            var scaleShape = GetBufferShape(scale);
            var biasShape = GetBufferShape(bias);
            var outputShape = GetBufferShape(output);
            var normalizedAxis = NormalizeAxis(layerNorm.Axis, outputShape.Length, "PyNTT LayerNorm");
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT LayerNorm input");
            var scaleVectorLaneCount = GetSingleVectorLaneCount(scale.ElemType, "PyNTT LayerNorm scale");
            var biasVectorLaneCount = GetSingleVectorLaneCount(bias.ElemType, "PyNTT LayerNorm bias");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT LayerNorm output");
            var valueLaneCounts = new[] { inputVectorLaneCount, scaleVectorLaneCount, biasVectorLaneCount, outputVectorLaneCount }
                .Where(lane => lane > 1)
                .Distinct()
                .ToArray();
            if (valueLaneCounts.Length > 1)
            {
                throw new NotSupportedException($"PyNTT LayerNorm expects matching input/scale/bias/output vector lanes, got input={inputVectorLaneCount}, scale={scaleVectorLaneCount}, bias={biasVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            if (valueLaneCounts.Length == 1)
            {
                var vectorizedAxes = layerNorm.VectorizedAxes.IsDefaultOrEmpty ? Array.Empty<int>() : layerNorm.VectorizedAxes.ToArray();
                if (vectorizedAxes.Length != 1 || NormalizeAxis(vectorizedAxes[0], outputShape.Length, "PyNTT LayerNorm vectorized axis") != outputShape.Length - 1)
                {
                    throw new NotSupportedException($"PyNTT LayerNorm currently supports vectorization only on the last normalized axis, got [{string.Join(",", vectorizedAxes)}].");
                }

                if (normalizedAxis > outputShape.Length - 1)
                {
                    throw new NotSupportedException("PyNTT LayerNorm vectorized axis must be inside the normalized dimensions.");
                }
            }

            var logicalInputShape = GetLogicalVectorShape(inputShape, inputVectorLaneCount);
            var logicalScaleShape = GetLogicalVectorShape(scaleShape, scaleVectorLaneCount);
            var logicalBiasShape = GetLogicalVectorShape(biasShape, biasVectorLaneCount);
            var logicalOutputShape = GetLogicalVectorShape(outputShape, outputVectorLaneCount);
            ValidateSameShape("PyNTT LayerNorm", logicalInputShape, logicalOutputShape);
            ValidateLayerNormShape("PyNTT LayerNorm scale", logicalScaleShape, logicalOutputShape, normalizedAxis);
            ValidateLayerNormShape("PyNTT LayerNorm bias", logicalBiasShape, logicalOutputShape, normalizedAxis);
            if ((GetShardAxis(input) is int inputShardAxis && inputShardAxis >= normalizedAxis) ||
                (GetShardAxis(output) is int outputShardAxis && outputShardAxis >= normalizedAxis) ||
                GetShardAxis(scale).HasValue ||
                GetShardAxis(bias).HasValue)
            {
                throw new NotSupportedException("PyNTT LayerNorm codegen does not support sharding along normalized dimensions or sharded scale/bias yet.");
            }

            _attrs["op"] = layerNorm.UseMean ? "layer_norm" : "rms_norm";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = logicalOutputShape;
            _attrs["axis"] = normalizedAxis;
            _attrs["epsilon"] = layerNorm.Epsilon;
            _attrs["use_mean"] = layerNorm.UseMean;
            var helperName = GetNextHelperName("layer_norm_compute");
            WriteHelperTemplate(
                "triton/kernels/LayerNorm.py.jinja",
                new PyNTTLayerNormTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(scale),
                    GetBufferScalarPointer(bias),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(scale.ElemType),
                    GetPyNTTScalarDTypeName(bias.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(scale.ElemType),
                    GetScalarTritonDType(bias.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    scaleShape,
                    biasShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(scale),
                    GetBufferStrides(bias),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    scaleVectorLaneCount,
                    biasVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(input.ElemType),
                    GetVectorLanes(scale.ElemType),
                    GetVectorLanes(bias.ElemType),
                    GetVectorLanes(output.ElemType),
                    normalizedAxis,
                    layerNorm.Epsilon,
                    layerNorm.UseMean,
                    $"{input.Name}, {scale.Name}, {bias.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitNormStats(Nncase.TIR.NTT.NormStats normStats, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT NormStats codegen expects input and output TIR buffers.");
            }

            SetComputeOp(normStats.UseMean ? "norm_stats" : "rms_norm_stats");
            var inputShape = GetBufferActiveShape(input);
            var outputShape = GetBufferActiveShape(output);
            var normalizedAxis = NormalizeAxis(normStats.Axis, inputShape.Length, "PyNTT NormStats");
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT NormStats input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT NormStats output");
            if (outputVectorLaneCount != 1)
            {
                throw new NotSupportedException("PyNTT NormStats expects scalar stats output buffer dtype.");
            }

            ValidateNormStatsShape("PyNTT NormStats", inputShape, outputShape, normalizedAxis, normStats.UseMean);

            _attrs["op"] = normStats.UseMean ? "norm_stats" : "rms_norm_stats";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axis"] = normalizedAxis;
            _attrs["use_mean"] = normStats.UseMean;
            var helperName = GetNextHelperName("norm_stats_compute");
            WriteHelperTemplate(
                "triton/kernels/NormStats.py.jinja",
                new PyNTTNormStatsTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(input.ElemType),
                    GetVectorLanes(output.ElemType),
                    normalizedAxis,
                    normStats.UseMean,
                    $"{input.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
        }

        private void VisitNormApply(Nncase.TIR.NTT.NormApply normApply, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 5)
            {
                throw new NotSupportedException("PyNTT NormApply codegen expects input, stats, scale, bias, and output TIR buffers.");
            }

            var input = GetBufferOperand(args[0], "PyNTT NormApply input");
            var stats = GetBufferOperand(args[1], "PyNTT NormApply stats");
            var scale = GetBufferOperand(args[2], "PyNTT NormApply scale");
            var bias = GetBufferOperand(args[3], "PyNTT NormApply bias");
            var output = GetBufferOperand(args[4], "PyNTT NormApply output");
            SetComputeOp(normApply.UseMean ? "norm_apply" : "rms_norm_apply");
            var inputShape = GetBufferActiveShape(input);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var statsShape = GetBufferActiveShape(stats);
            var scaleShape = GetBufferActiveShape(scale);
            var biasShape = GetBufferActiveShape(bias);
            var outputShape = GetBufferActiveShape(output);
            var normalizedAxis = NormalizeAxis(normApply.Axis, outputShape.Length, "PyNTT NormApply");
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT NormApply input");
            var statsVectorLaneCount = GetSingleVectorLaneCount(stats.ElemType, "PyNTT NormApply stats");
            var scaleVectorLaneCount = GetSingleVectorLaneCount(scale.ElemType, "PyNTT NormApply scale");
            var biasVectorLaneCount = GetSingleVectorLaneCount(bias.ElemType, "PyNTT NormApply bias");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT NormApply output");
            if (statsVectorLaneCount != 1)
            {
                throw new NotSupportedException("PyNTT NormApply expects scalar stats buffer dtype.");
            }

            if (new[] { inputVectorLaneCount, scaleVectorLaneCount, biasVectorLaneCount, outputVectorLaneCount }.Distinct().Count() != 1)
            {
                throw new NotSupportedException($"PyNTT NormApply expects matching input/scale/bias/output vector lanes, got input={inputVectorLaneCount}, scale={scaleVectorLaneCount}, bias={biasVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            if (inputVectorLaneCount != 1 && normalizedAxis > outputShape.Length - 1)
            {
                throw new NotSupportedException("PyNTT NormApply vectorized axis must be inside the normalized dimensions.");
            }

            var logicalInputShape = GetLogicalVectorShape(inputShape, inputVectorLaneCount);
            var logicalStatsShape = GetLogicalVectorShape(statsShape, statsVectorLaneCount);
            var logicalScaleShape = GetLogicalVectorShape(scaleShape, scaleVectorLaneCount);
            var logicalBiasShape = GetLogicalVectorShape(biasShape, biasVectorLaneCount);
            var logicalOutputShape = GetLogicalVectorShape(outputShape, outputVectorLaneCount);
            ValidateSameShape("PyNTT NormApply", logicalInputShape, logicalOutputShape);
            ValidateNormStatsShape("PyNTT NormApply stats", logicalOutputShape, logicalStatsShape, normalizedAxis, normApply.UseMean);
            ValidateLayerNormShape("PyNTT NormApply scale", logicalScaleShape, logicalOutputShape, normalizedAxis);
            ValidateLayerNormShape("PyNTT NormApply bias", logicalBiasShape, logicalOutputShape, normalizedAxis);
            _attrs["op"] = normApply.UseMean ? "norm_apply" : "rms_norm_apply";
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = logicalOutputShape;
            _attrs["axis"] = normalizedAxis;
            _attrs["epsilon"] = normApply.Epsilon;
            _attrs["use_mean"] = normApply.UseMean;
            var helperName = GetNextHelperName("norm_apply_compute");
            WriteHelperTemplate(
                "triton/kernels/NormApply.py.jinja",
                new PyNTTNormApplyTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(stats),
                    GetBufferScalarPointer(scale),
                    GetBufferScalarPointer(bias),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(stats.ElemType),
                    GetPyNTTScalarDTypeName(scale.ElemType),
                    GetPyNTTScalarDTypeName(bias.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(stats.ElemType),
                    GetScalarTritonDType(scale.ElemType),
                    GetScalarTritonDType(bias.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    inputGlobalShape,
                    statsShape,
                    scaleShape,
                    biasShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(stats),
                    GetBufferStrides(scale),
                    GetBufferStrides(bias),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    statsVectorLaneCount,
                    scaleVectorLaneCount,
                    biasVectorLaneCount,
                    outputVectorLaneCount,
                    GetVectorLanes(input.ElemType),
                    GetVectorLanes(stats.ElemType),
                    GetVectorLanes(scale.ElemType),
                    GetVectorLanes(bias.ElemType),
                    GetVectorLanes(output.ElemType),
                    normalizedAxis,
                    normApply.Epsilon,
                    normApply.UseMean,
                    $"{input.Name}, {stats.Name}, {scale.Name}, {bias.Name} -> {output.Name}"));
            WriteHelperInvocation(helperName);
            MarkStoredOutput(output, "PyNTT NormApply");
        }

        private void VisitGetPositionIds(Nncase.TIR.NTT.GetPositionIds getPositionIds, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException($"PyNTT GetPositionIds codegen expects kv-cache input and output TIR buffer, got ({string.Join(", ", args.Select(arg => arg.GetType().Name))}).");
            }

            var outputShape = GetBufferShape(output);
            ValidateRank("PyNTT GetPositionIds output", outputShape, 1);
            var globalShape = GetRankedShapeDimensions(getPositionIds.DistributedType.TensorType.Shape, "PyNTT GetPositionIds global shape")
                .Select(GetDimensionExpression)
                .ToArray();
            ValidateRank("PyNTT GetPositionIds global output", globalShape, 1);
            var cacheMetaArgument = GetKVCacheFieldArgument(args[0], "metadata");
            SetComputeOp("get_position_ids");
            _attrs["op"] = "get_position_ids";
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            var helperName = GetNextHelperName("get_position_ids_compute");
            WriteHelperTemplate(
                "triton/kernels/GetPositionIds.py.jinja",
                new PyNTTGetPositionIdsTemplateModel(
                    helperName,
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(output.ElemType),
                    outputShape,
                    globalShape,
                    GetBufferGlobalOffsets(output),
                    GetBufferStrides(output),
                    $"kv-cache -> {output.Name}"));
            WriteHelperInvocation(helperName, cacheMetaArgument);
        }

        private void VisitUpdatePagedAttentionKVCache(Nncase.TIR.NTT.UpdatePagedAttentionKVCache update, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer slots)
            {
                throw new NotSupportedException("PyNTT UpdatePagedAttentionKVCache codegen expects slots buffer, kv-cache object, and layer id.");
            }

            var cache = GetPagedAttentionCacheTemplateModel(args[1], "PyNTT UpdatePagedAttentionKVCache");
            var layerIdExpression = GetDimensionExpression(args[2], "PyNTT UpdatePagedAttentionKVCache layer id").TritonExpression;
            var storage = GetKVCacheStorageMetadata(cache);
            var metaArgument = GetKVCacheFieldArgument(args[1], "metadata");
            var slotMappingArgument = GetKVCacheFieldArgument(args[1], "slot_mapping");
            var storageArgument = GetKVCacheFieldArgument(args[1], "kv_caches", storage);
            var storageBlocksArgument = GetKVCacheFieldArgument(args[1], "kv_caches_blocks", storage);
            var layout = update.Layout.ToArray();
            var seqAxis = Array.IndexOf(layout, AttentionDimKind.Seq);
            var headAxis = Array.IndexOf(layout, AttentionDimKind.Head);
            var dimAxis = Array.IndexOf(layout, AttentionDimKind.Dim);
            if (seqAxis < 0 || headAxis < 0 || dimAxis < 0)
            {
                throw new NotSupportedException("PyNTT UpdatePagedAttentionKVCache layout must contain Seq, Head, and Dim.");
            }

            SetComputeOp("update_paged_attention_kv_cache");
            _attrs["op"] = "update_paged_attention_kv_cache";
            _attrs["cache_kind"] = update.CacheKind.ToString();
            _attrs["layer_id"] = layerIdExpression;
            var helperName = GetNextHelperName("update_paged_attention_kv_cache");
            var slotsOperand = GetBufferScalarPointer(slots);
            WriteHelperTemplate(
                "triton/kernels/UpdatePagedAttentionKVCache.py.jinja",
                new PyNTTUpdatePagedAttentionKVCacheTemplateModel(
                    helperName,
                    slotsOperand,
                    GetPyNTTScalarDTypeName(slots.ElemType),
                    GetScalarTritonDType(slots.ElemType),
                    GetBufferShape(slots),
                    GetBufferGlobalShape(slots),
                    GetBufferGlobalOffsets(slots),
                    GetBufferStrides(slots),
                    GetBufferSplitAxes(slots, slots.Dimensions.Length),
                    GetBufferSourceSplitAxes(slots, slots.Dimensions.Length),
                    GetHierarchy(slots),
                    seqAxis,
                    headAxis,
                    dimAxis,
                    layerIdExpression,
                    update.CacheKind == AttentionCacheKind.Key ? 0 : 1,
                    GetVectorLaneElementCount(slots.ElemType),
                    GetVectorLanes(slots.ElemType),
                    cache,
                    $"{slots.Name} -> kv-cache"));
            WriteLine(BuildHelperCall(helperName, slotMappingArgument, storageArgument, storageBlocksArgument, metaArgument));
        }

        private void VisitPagedAttention(Nncase.TIR.NTT.PagedAttention pagedAttention, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 6 ||
                args[0] is not TIR.Buffer query ||
                args[2] is not TIR.Buffer ||
                args[3] is not TIR.Buffer scale ||
                args[5] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT PagedAttention codegen expects query, kv-cache, extra, scale, layer id, and output buffers.");
            }

            var cache = GetPagedAttentionCacheTemplateModel(args[1], "PyNTT PagedAttention");
            var layerIdExpression = GetDimensionExpression(args[4], "PyNTT PagedAttention layer id").TritonExpression;
            var storage = GetKVCacheStorageMetadata(cache);
            var metaArgument = GetKVCacheFieldArgument(args[1], "metadata");
            var blockTablesArgument = GetKVCacheFieldArgument(args[1], "block_tables");
            var storageArgument = GetKVCacheFieldArgument(args[1], "kv_caches", storage);
            var storageBlocksArgument = GetKVCacheFieldArgument(args[1], "kv_caches_blocks", storage);
            var layout = pagedAttention.Layout.ToArray();
            var seqAxis = Array.IndexOf(layout, AttentionDimKind.Seq);
            var headAxis = Array.IndexOf(layout, AttentionDimKind.Head);
            var dimAxis = Array.IndexOf(layout, AttentionDimKind.Dim);
            if (seqAxis < 0 || headAxis < 0 || dimAxis < 0)
            {
                throw new NotSupportedException("PyNTT PagedAttention layout must contain Seq, Head, and Dim.");
            }

            var querySplitAxes = GetBufferSplitAxes(query, query.Dimensions.Length);
            var outputSplitAxes = GetBufferSplitAxes(output, output.Dimensions.Length);
            if (querySplitAxes[dimAxis].Length != 0 || outputSplitAxes[dimAxis].Length != 0)
            {
                throw new NotSupportedException("PyNTT PagedAttention codegen requires the attention Dim axis to be unsplit.");
            }

            SetComputeOp("paged_attention");
            _attrs["op"] = "paged_attention";
            _attrs["layer_id"] = layerIdExpression;
            _attrs["hidden_size"] = pagedAttention.HiddenSize;
            var helperName = GetNextHelperName("paged_attention");
            WriteHelperTemplate(
                "triton/kernels/PagedAttention.py.jinja",
                new PyNTTPagedAttentionTemplateModel(
                    helperName,
                    GetBufferScalarPointer(query),
                    GetBufferScalarPointer(scale),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(query.ElemType),
                    GetScalarTritonDType(query.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    GetBufferShape(query),
                    GetBufferShape(output),
                    GetBufferGlobalShape(output),
                    GetBufferStrides(query),
                    GetBufferStrides(output),
                    GetVectorLanes(query.ElemType),
                    GetVectorLanes(output.ElemType),
                    outputSplitAxes,
                    GetHierarchy(output),
                    seqAxis,
                    headAxis,
                    dimAxis,
                    GetGlobalNumQueryHeads(pagedAttention, cache),
                    layerIdExpression,
                    _targetOptions.TargetMachineModel.Execution.WorkerWidth,
                    cache,
                    $"{query.Name}, kv-cache -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName, blockTablesArgument, storageArgument, storageBlocksArgument, metaArgument));
        }

        private void VisitSoftmax(int axis, IRArray<int> vectorizedAxes, IReadOnlyList<BaseExpr> args, string opKind)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Softmax codegen expects input and output TIR buffers.");
            }

            EnsureEmpty("PyNTT Softmax vectorized axes", vectorizedAxes);
            SetComputeOp(opKind);
            var shape = GetBufferShape(output);
            ValidateSameShape("PyNTT Softmax", GetBufferShape(input), shape);
            var normalizedAxis = NormalizeAxis(axis, shape.Length, "PyNTT Softmax");
            if (GetShardAxis(output) == normalizedAxis || GetShardAxis(input) == normalizedAxis)
            {
                throw new NotSupportedException("PyNTT Softmax codegen does not support sharding along the softmax axis yet.");
            }

            _attrs["op"] = "softmax";
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["axis"] = normalizedAxis;
            var helperName = GetNextHelperName("softmax_compute");
            WriteHelperTemplate(
                "triton/kernels/Softmax.py.jinja",
                new PyNTTSoftmaxTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    normalizedAxis,
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private int GetInputIndex(BaseExpr expr)
        {
            expr = ResolveBoundExpression(expr);
            if (TryGetFormalObjectBaseName(expr, out var formalObjectBaseName))
            {
                throw new NotSupportedException($"PyNTT formal object parameter {formalObjectBaseName} cannot be registered as a kernel tensor input. Use a concrete object field argument instead.");
            }

            var inputName = GetTensorName(expr, _parameterNames);
            var inputIndex = _inputNames.IndexOf(inputName);
            if (inputIndex < 0)
            {
                inputIndex = _inputNames.Count;
                _inputNames.Add(inputName);
            }

            return inputIndex;
        }

        private TIR.Buffer GetBufferOperand(BaseExpr expr, string context)
        {
            expr = ResolveBoundExpression(expr);
            return expr switch
            {
                TIR.Buffer buffer => buffer,
                BufferVar bufferVar => GetAbiBuffer(bufferVar, context),
                Call { Target: Nncase.IR.Buffers.AllocateBufferView } allocate => MaterializeAllocateBufferView($"{context}_view", allocate),
                Call { Target: Nncase.IR.Buffers.BufferSubview } subview => MaterializeBufferSubview($"{context}_view", subview),
                _ => throw new NotSupportedException($"{context} expects a TIR buffer or buffer ABI parameter, got {expr.GetType().Name}."),
            };
        }

        private bool TryGetFormalTensorBuffer(BaseExpr expr, string context, out TIR.Buffer buffer)
        {
            expr = ResolveBoundExpression(expr);
            if (expr is BufferVar bufferVar && _formalTensorParameterBaseNames.ContainsKey(bufferVar))
            {
                buffer = GetAbiBuffer(bufferVar, context);
                return true;
            }

            if (expr is TIR.Buffer candidate && IsFormalTensorBuffer(candidate))
            {
                buffer = candidate;
                return true;
            }

            buffer = null!;
            return false;
        }

        private BaseExpr ResolveBoundExpression(BaseExpr expr)
        {
            expr = UnwrapInputBoxing(expr);
            var seen = new HashSet<IVar>(ReferenceEqualityComparer.Instance);
            while (expr is IVar var && _letBindings.TryGetValue(var, out var value))
            {
                if (!seen.Add(var))
                {
                    throw new NotSupportedException($"PyNTT Let bindings contain a cycle at {var.Name}.");
                }

                expr = UnwrapInputBoxing(value);
            }

            return expr switch
            {
                Call { Target: Nncase.IR.Buffers.AllocateBufferView } allocate => MaterializeAllocateBufferView("pyntt_buffer_view", allocate),
                Call { Target: Nncase.IR.Buffers.BufferSubview } subview => MaterializeBufferSubview("pyntt_subview", subview),
                _ => expr,
            };
        }

        private BaseExpr ResolveCallArgument(BaseExpr expr, int index)
        {
            try
            {
                return ResolveBoundExpression(expr);
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
            {
                throw new NotSupportedException($"PyNTT failed to resolve call argument {index}: {ex.Message}", ex);
            }
        }

        private BaseExpr MaterializeLetBinding(IVar var, BaseExpr value)
        {
            value = UnwrapInputBoxing(value);
            return value switch
            {
                Call { Target: Nncase.IR.Buffers.AllocateBufferView } allocate => MaterializeAllocateBufferView(var.Name, allocate),
                Call { Target: Nncase.IR.Buffers.BufferSubview } subview => MaterializeBufferSubview(var.Name, subview),
                _ => ResolveBoundExpression(value),
            };
        }

        private TIR.Buffer MaterializeAllocateBufferView(string name, Call call)
        {
            var args = call.Arguments.ToArray();
            if (args.Length != 2)
            {
                throw new NotSupportedException($"PyNTT AllocateBufferView codegen expects a buffer and its local logical offset, got {args.Length} arguments.");
            }

            var source = GetBufferOperand(args[0], "PyNTT AllocateBufferView source");
            var localOffsets = GetShapeArgument(args[1], "PyNTT AllocateBufferView local offset")
                .Select(GetDimensionExpression)
                .ToArray();
            if (localOffsets.Length != source.Rank)
            {
                throw new NotSupportedException($"PyNTT AllocateBufferView rank mismatch: source rank {source.Rank}, local offset rank {localOffsets.Length}.");
            }

            var sourceGlobalOffsets = GetBufferGlobalOffsets(source);
            var globalOffsets = sourceGlobalOffsets
                .Zip(localOffsets)
                .Select(pair => AddDimExpression(pair.First, pair.Second))
                .ToArray();
            var activeShape = source.Dimensions.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
            var result = source.With(name: SanitizePythonIdentifier(name));
            _bufferActiveShapeOverrides[result] = activeShape;
            _bufferGlobalShapeOverrides[result] = GetBufferGlobalShape(source);
            _bufferGlobalOffsetOverrides[result] = globalOffsets;
            _bufferSourceSplitAxesOverrides[result] = GetBufferSourceSplitAxes(source, source.Rank);
            _bufferViewSourceByBuffer[result] = new(source, CreateZeroDimExpressions(source.Rank), activeShape);
            TrackObjectViewSource(result, source);
            TrackObjectViewAlias(result, source);
            return result;
        }

        private SharedBufferAllocation EmitSharedBufferAllocation(IVar variable, TIR.Buffer buffer)
        {
            var scalarElementSizeBytes = GetScalarElementSizeBytes(buffer.ElemType);
            var physicalOffsetBytes = GetFixedDimension(
                buffer.MemSpan.Buffer.Start,
                $"PyNTT shared buffer {buffer.Name} physical offset");
            var physicalSizeBytes = GetFixedDimension(
                buffer.MemSpan.Buffer.Size,
                $"PyNTT shared buffer {buffer.Name} physical size");
            var spanOffsetBytes = GetFixedDimension(
                buffer.MemSpan.Start,
                $"PyNTT shared buffer {buffer.Name} span offset");
            var spanSizeBytes = GetFixedDimension(
                buffer.MemSpan.Size,
                $"PyNTT shared buffer {buffer.Name} span size");
            if (physicalOffsetBytes < 0 ||
                physicalSizeBytes <= 0 ||
                spanOffsetBytes < 0 ||
                spanSizeBytes <= 0 ||
                checked(spanOffsetBytes + spanSizeBytes) > physicalSizeBytes)
            {
                throw new NotSupportedException(
                    $"PyNTT shared AllocateBufferView {variable.Name} has invalid physical range " +
                    $"[{physicalOffsetBytes}, {checked(physicalOffsetBytes + physicalSizeBytes)}) and span " +
                    $"[{spanOffsetBytes}, {checked(spanOffsetBytes + spanSizeBytes)}).");
            }

            var aliasOffsetBytes = checked(physicalOffsetBytes + spanOffsetBytes);
            var storageEncoding = buffer.StorageEncoding;
            var storageAlignmentBytes = storageEncoding?.AlignmentBytes ?? scalarElementSizeBytes;
            if (storageEncoding is not null && storageEncoding.PhysicalBytes != physicalSizeBytes)
            {
                throw new NotSupportedException(
                    $"PyNTT shared buffer {buffer.Name} carries storage encoding {storageEncoding} with " +
                    $"{storageEncoding.PhysicalBytes} physical bytes, but its TIR PhysicalBuffer has {physicalSizeBytes} bytes.");
            }

            if (scalarElementSizeBytes <= 0 ||
                aliasOffsetBytes % storageAlignmentBytes != 0 ||
                spanSizeBytes % scalarElementSizeBytes != 0)
            {
                throw new NotSupportedException(
                    $"PyNTT shared AllocateBufferView {variable.Name} range " +
                    $"[{aliasOffsetBytes}, {checked(aliasOffsetBytes + spanSizeBytes)}) is not aligned " +
                    $"to storage alignment {storageAlignmentBytes} and scalar element size {scalarElementSizeBytes}.");
            }

            var elementCount = checked((ulong)(spanSizeBytes / scalarElementSizeBytes));
            var descriptorCapacityElements = checked((ulong)((physicalSizeBytes - spanOffsetBytes) / scalarElementSizeBytes));
            if (elementCount == 0 || elementCount > long.MaxValue)
            {
                throw new NotSupportedException(
                    $"PyNTT shared AllocateBufferView {variable.Name} with {elementCount} elements " +
                    "cannot be represented as a Triton memdesc.");
            }

            var storageEncodingId = storageEncoding?.Id ?? TargetStorageEncodingIds.Linear;
            var useNvidiaMmaSharedLayout = storageEncodingId == TritonTargetStorageEncodingModel.NvidiaMmaShared;
            var useKMajorPackedNLayout = storageEncodingId == TritonTargetStorageEncodingModel.KMajorPackedN;
            var logicalShape = GetBufferShape(buffer)
                .Select((dimension, axis) => dimension.MaxValue is { } value && value > 0
                    ? value
                    : throw new NotSupportedException(
                        $"PyNTT shared AllocateBufferView {variable.Name} dimension {axis} must have a positive static maximum, " +
                        $"got range [{dimension.MinValue}, {dimension.MaxValue}]."))
                .ToArray();
            var logicalStrides = buffer.Strides.ToArray()
                .Select((stride, axis) => stride.IsFixed &&
                    (stride.FixedValue > 0 || (stride.FixedValue == 0 && logicalShape[axis] == 1))
                    ? stride.FixedValue
                    : throw new NotSupportedException(
                        $"PyNTT shared AllocateBufferView {variable.Name} stride {axis} must be positive, " +
                        $"or zero only for a singleton dimension; shape={logicalShape[axis]}, stride={stride}."))
                .ToArray();
            var vectorLaneShape = GetVectorLanes(buffer.ElemType);
            var descriptorShape = GetSharedDescriptorShape(
                variable,
                buffer,
                checked((long)elementCount),
                checked((long)descriptorCapacityElements),
                storageEncodingId == TargetStorageEncodingIds.Linear ||
                    storageEncodingId == TritonTargetStorageEncodingModel.SwizzledShared,
                useNvidiaMmaSharedLayout,
                useKMajorPackedNLayout,
                logicalShape,
                logicalStrides,
                vectorLaneShape);
            var descriptorElementCount = descriptorShape.Aggregate(1L, (product, value) => checked(product * value));
            var descriptorSizeBytes = checked(descriptorElementCount * scalarElementSizeBytes);
            var descriptorEndBytes = checked(aliasOffsetBytes + descriptorSizeBytes);
            _sharedAliasRequiredBytes = Math.Max(_sharedAliasRequiredBytes, descriptorEndBytes);
            var descriptorName = SanitizeBoundedPythonIdentifier(
                $"{variable.Name}_shared_buffer_{_sharedBufferAllocationCounter++.ToString(CultureInfo.InvariantCulture)}");
            var tritonDType = GetScalarTritonDType(buffer.ElemType);
            if (storageEncodingId != TargetStorageEncodingIds.Linear &&
                storageEncodingId != TritonTargetStorageEncodingModel.SwizzledShared &&
                !useNvidiaMmaSharedLayout &&
                !useKMajorPackedNLayout)
            {
                throw new NotSupportedException(
                    $"PyNTT Triton shared allocation does not support storage encoding {storageEncodingId} for {buffer.Name}.");
            }

            WriteControlLine(
                $"{descriptorName} = tle.gpu.alloc([{string.Join(", ", descriptorShape.Select(value => value.ToString(CultureInfo.InvariantCulture)))}], " +
                $"dtype={tritonDType}, layout=None, scope=tle.gpu.smem, alias=pyntt_shared_arena, " +
                $"alias_offset_bytes={aliasOffsetBytes.ToString(CultureInfo.InvariantCulture)}, " +
                $"nv_mma_shared_layout={(useNvidiaMmaSharedLayout ? "True" : "False")})");
            return new(
                descriptorName,
                descriptorShape,
                logicalShape,
                logicalStrides,
                vectorLaneShape,
                checked(physicalSizeBytes - spanOffsetBytes),
                scalarElementSizeBytes,
                tritonDType,
                storageEncodingId.Value);
        }

        private long[] GetSharedDescriptorShape(
            IVar variable,
            TIR.Buffer buffer,
            long elementCount,
            long descriptorCapacityElements,
            bool allowAffineLinearDescriptor,
            bool useNvidiaMmaSharedLayout,
            bool useKMajorPackedNLayout,
            IReadOnlyList<long> logicalShape,
            IReadOnlyList<long> logicalStrides,
            IReadOnlyList<int> vectorLaneShape)
        {
            if (logicalShape.Count == 0)
            {
                throw new NotSupportedException(
                    $"PyNTT shared AllocateBufferView {variable.Name} has a rank-zero TIR Buffer. " +
                    "FlagTree TLE requires a non-empty memdesc shape.");
            }

            // TLE descriptors use scalar dtypes, so vector lanes remain explicit trailing physical dimensions.
            var scalarPhysicalShape = logicalShape
                .Concat(vectorLaneShape.Select(lane => checked((long)lane)))
                .ToArray();
            if (logicalStrides.Count != logicalShape.Count)
            {
                throw new NotSupportedException(
                    $"PyNTT shared AllocateBufferView {variable.Name} shape/stride rank mismatch: " +
                    $"shape={logicalShape.Count}, strides={logicalStrides.Count}.");
            }

            var vectorLaneCount = GetVectorLaneElementCount(buffer.ElemType);
            var descriptorLogicalStrides = logicalStrides.ToArray();
            for (var axis = descriptorLogicalStrides.Length - 1; axis >= 0; axis--)
            {
                if (descriptorLogicalStrides[axis] != 0)
                {
                    continue;
                }

                descriptorLogicalStrides[axis] = axis == descriptorLogicalStrides.Length - 1
                    ? 1
                    : checked(descriptorLogicalStrides[axis + 1] * logicalShape[axis + 1]);
            }

            var requiredScalarElements = checked((long)vectorLaneCount);
            for (var axis = 0; axis < logicalShape.Count; axis++)
            {
                requiredScalarElements = checked(
                    requiredScalarElements +
                    ((logicalShape[axis] - 1) * logicalStrides[axis] * vectorLaneCount));
            }

            if (requiredScalarElements > elementCount)
            {
                throw new NotSupportedException(
                    $"PyNTT shared AllocateBufferView {variable.Name} shape " +
                    $"[{string.Join(",", scalarPhysicalShape)}] and strides [{string.Join(",", logicalStrides)}] " +
                    $"can address {requiredScalarElements} scalar elements, but its TIR MemSpan capacity is only {elementCount}.");
            }

            // Packed GEMV stores weights logically as [G,K]vec<lanes>. Keep
            // that grouped physical structure in shared memory so the backend
            // can reduce K without transposing or flattening G.
            if (useKMajorPackedNLayout)
            {
                if (logicalShape.Count != 2 || vectorLaneShape.Count == 0)
                {
                    throw new NotSupportedException(
                        $"PyNTT K-major packed-N shared buffer {variable.Name} must be a rank-2 vector buffer, " +
                        $"got rank {logicalShape.Count} and vector-lane shape [{string.Join(",", vectorLaneShape)}].");
                }

                if (descriptorLogicalStrides[1] != 1)
                {
                    throw new NotSupportedException(
                        $"PyNTT K-major packed-N shared buffer {variable.Name} requires a contiguous K axis " +
                        $"logical storage, got shape [{string.Join(",", logicalShape)}] and strides " +
                        $"[{string.Join(",", logicalStrides)}].");
                }

                var packedDescriptorShape = new[]
                {
                    logicalShape[0],
                    logicalShape[1],
                    checked((long)vectorLaneCount),
                };
                var packedDescriptorElements = packedDescriptorShape.Aggregate(
                    1L,
                    (product, value) => checked(product * value));
                if (packedDescriptorElements > descriptorCapacityElements)
                {
                    throw new NotSupportedException(
                        $"PyNTT K-major packed-N shared descriptor for {variable.Name} requires " +
                        $"{packedDescriptorElements} scalar elements, but only {descriptorCapacityElements} remain " +
                        "in its TIR PhysicalBuffer allocation.");
                }

                return packedDescriptorShape;
            }

            // Packed matrix weights use [N / lanes, K]vec<lanes> in TIR. A
            // single packed-N group is physically identical to the [K, N]
            // rectangle consumed by Triton dot. Expose that rectangle as the
            // memdesc shape so FlagTree can lower a full-view load directly to
            // the selected MMA shared allocation instead of staging it again.
            if (useNvidiaMmaSharedLayout && vectorLaneShape.Count > 0)
            {
                if (logicalShape.Count != 2)
                {
                    throw new NotSupportedException(
                        $"PyNTT NVIDIA MMA shared vector buffer {variable.Name} must have rank 2, got rank {logicalShape.Count}.");
                }

                var packedGroupCount = scalarPhysicalShape[0];
                if (packedGroupCount == 1)
                {
                    var packedDescriptorShape = new[] { scalarPhysicalShape[1], checked((long)vectorLaneCount) };
                    var packedDescriptorElementCount = checked(packedDescriptorShape[0] * packedDescriptorShape[1]);
                    if (packedDescriptorElementCount > elementCount)
                    {
                        throw new NotSupportedException(
                            $"PyNTT NVIDIA MMA shared descriptor for {variable.Name} requires {packedDescriptorElementCount} " +
                            $"scalar elements, but its TIR MemSpan contains only {elementCount}.");
                    }

                    return packedDescriptorShape;
                }
            }

            // A linear shared descriptor is the rectangular physical view of
            // Buffer.Dimensions/Strides. Keep every logical and vector-lane
            // axis explicit; do not flatten and later recover coordinates.
            var descriptorShape = scalarPhysicalShape.ToArray();
            var expectedInnerStride = 1L;
            for (var axis = descriptorShape.Length - 1; axis >= logicalShape.Count; axis--)
            {
                expectedInnerStride = checked(expectedInnerStride * descriptorShape[axis]);
            }

            if (checked(descriptorLogicalStrides[^1] * vectorLaneCount) != expectedInnerStride)
            {
                throw new NotSupportedException(
                    $"PyNTT shared buffer {variable.Name} innermost scalar stride " +
                    $"{checked(descriptorLogicalStrides[^1] * vectorLaneCount)} cannot be represented by the explicit " +
                    $"vector-lane shape [{string.Join(",", vectorLaneShape)}].");
            }

            for (var axis = logicalShape.Count - 2; axis >= 0; axis--)
            {
                var outerStride = descriptorLogicalStrides[axis];
                var innerStride = descriptorLogicalStrides[axis + 1];
                if (outerStride % innerStride != 0)
                {
                    throw new NotSupportedException(
                        $"PyNTT shared buffer {variable.Name} strides [{string.Join(",", logicalStrides)}] " +
                        "do not describe a rectangular strided view.");
                }

                var physicalInnerExtent = outerStride / innerStride;
                if (physicalInnerExtent < logicalShape[axis + 1])
                {
                    throw new NotSupportedException(
                        $"PyNTT shared buffer {variable.Name} stride {outerStride} overlaps logical axis " +
                        $"{axis + 1} with extent {logicalShape[axis + 1]} and stride {innerStride}.");
                }

                descriptorShape[axis + 1] = physicalInnerExtent;
            }

            var descriptorElementCount = descriptorShape.Aggregate(1L, (product, value) => checked(product * value));
            if (requiredScalarElements > descriptorElementCount)
            {
                var scalarOuterStride = checked(descriptorLogicalStrides[0] * vectorLaneCount);
                descriptorShape[0] = Math.Max(
                    descriptorShape[0],
                    checked((requiredScalarElements + scalarOuterStride - 1) / scalarOuterStride));
            }

            var hasNonPowerOfTwoInnerExtent = false;
            for (var axis = 1; axis < descriptorShape.Length; axis++)
            {
                if (!System.Numerics.BitOperations.IsPow2((ulong)descriptorShape[axis]))
                {
                    hasNonPowerOfTwoInnerExtent = true;
                    break;
                }
            }

            if (hasNonPowerOfTwoInnerExtent)
            {
                if (!allowAffineLinearDescriptor)
                {
                    throw new NotSupportedException(
                        $"PyNTT shared buffer {variable.Name} with storage encoding {buffer.StorageEncoding} has " +
                        $"non-power-of-two inner descriptor shape [{string.Join(",", descriptorShape)}]. " +
                        "The selected target encoding cannot use the affine one-dimensional descriptor representation.");
                }

                var linearExtent = System.Numerics.BitOperations.RoundUpToPowerOf2((ulong)requiredScalarElements);
                if (linearExtent == 0 || linearExtent > (ulong)descriptorCapacityElements)
                {
                    throw new NotSupportedException(
                        $"PyNTT shared buffer {variable.Name} requires a {linearExtent}-element linear descriptor " +
                        $"for logical shape [{string.Join(",", logicalShape)}] and strides [{string.Join(",", logicalStrides)}], " +
                        $"but only {descriptorCapacityElements} scalar elements remain in its TIR PhysicalBuffer allocation.");
                }

                return new[] { checked((long)linearExtent) };
            }

            var paddedOuterExtent = System.Numerics.BitOperations.RoundUpToPowerOf2((ulong)descriptorShape[0]);
            if (paddedOuterExtent == 0 || paddedOuterExtent > long.MaxValue)
            {
                throw new NotSupportedException(
                    $"PyNTT shared buffer {variable.Name} outer extent {descriptorShape[0]} cannot be represented " +
                    "as a FlagTree block descriptor dimension.");
            }

            descriptorShape[0] = checked((long)paddedOuterExtent);

            var paddedDescriptorElementCount = descriptorShape.Aggregate(1L, (product, value) => checked(product * value));
            if (paddedDescriptorElementCount > descriptorCapacityElements)
            {
                throw new NotSupportedException(
                    $"PyNTT shared descriptor for {variable.Name} requires {paddedDescriptorElementCount} scalar elements, " +
                    $"but only {descriptorCapacityElements} remain in its TIR PhysicalBuffer allocation.");
            }

            return descriptorShape;
        }

        private TIR.Buffer MaterializeBufferSubview(string name, Call call)
        {
            var args = call.Arguments.ToArray();
            if (args.Length != 3)
            {
                throw new NotSupportedException($"PyNTT BufferSubview codegen expects 3 arguments, got {args.Length}.");
            }

            var source = GetBufferOperand(args[0], "PyNTT BufferSubview source");
            var offsets = GetShapeArgument(args[1], "PyNTT BufferSubview offsets");
            var shape = GetShapeArgument(args[2], "PyNTT BufferSubview shape");
            if (offsets.Length != source.Rank || shape.Length != source.Rank)
            {
                throw new NotSupportedException($"PyNTT BufferSubview rank mismatch: source rank {source.Rank}, offsets rank {offsets.Length}, shape rank {shape.Length}.");
            }

            var sourceGlobalOffsets = GetBufferGlobalOffsets(source);
            var offsetShape = offsets.Select(GetDimensionExpression).ToArray();
            var requestedShape = shape.Select(GetDimensionExpression).ToArray();
            var offsetElements = TensorUtilities.GetLinearOffset(source.Strides, offsets);
            var offsetBytes = offsetElements * source.ElemType.SizeInBytes;

            // A subview is already a concrete local region. Keep its global
            // distribution in the side tables below, but do not let local
            // kernel shape queries fall back to the source distributed type.
            var result = new TIR.Buffer(
                SanitizePythonIdentifier(name),
                source.ElemType,
                source.MemSpan.With(start: source.MemSpan.Start + offsetBytes),
                shape,
                source.Strides.ToArray(),
                null);
            _bufferActiveShapeOverrides[result] = requestedShape;
            _bufferGlobalShapeOverrides[result] = GetBufferGlobalShape(source);
            _bufferGlobalOffsetOverrides[result] = sourceGlobalOffsets.Zip(offsetShape).Select(pair => AddDimExpression(pair.First, pair.Second)).ToArray();
            _bufferSourceSplitAxesOverrides[result] = GetBufferSourceSplitAxes(source, source.Rank);
            _bufferViewSourceByBuffer[result] = new(source, offsetShape, requestedShape);
            TrackObjectViewSource(result, source);
            TrackObjectViewAlias(result, source);
            return result;
        }

        private void TrackObjectViewSource(TIR.Buffer dest, TIR.Buffer src)
        {
            if (!IsObjectDataType(dest.ElemType) && !IsObjectDataType(src.ElemType))
            {
                return;
            }

            _objectViewSourceByBuffer[dest] = src;
        }

        private void TrackObjectViewAlias(TIR.Buffer dest, TIR.Buffer src)
        {
            if (!IsObjectDataType(dest.ElemType) && !IsObjectDataType(src.ElemType))
            {
                return;
            }

            if (TryGetFormalObjectBaseName(src, out var formalObjectBaseName))
            {
                RecordObjectFormalAlias(dest, formalObjectBaseName);
                return;
            }

            if (TryResolveObjectInputIndex(src, out var inputIndex))
            {
                RecordObjectInputAlias(dest, inputIndex);
            }
        }

        private void RecordObjectFormalAlias(TIR.Buffer buffer, string formalObjectBaseName)
        {
            foreach (var aliasBuffer in EnumerateObjectAliasBuffers(buffer))
            {
                _formalObjectBaseNameByBuffer[aliasBuffer] = formalObjectBaseName;
            }
        }

        private void RecordObjectInputAlias(TIR.Buffer buffer, int inputIndex)
        {
            foreach (var aliasBuffer in EnumerateObjectAliasBuffers(buffer))
            {
                _bufferInputIndices[aliasBuffer] = inputIndex;
            }
        }

        private IEnumerable<TIR.Buffer> EnumerateObjectAliasBuffers(TIR.Buffer buffer)
        {
            var visited = new HashSet<TIR.Buffer>(ReferenceEqualityComparer.Instance);
            var current = buffer;
            while (visited.Add(current))
            {
                yield return current;
                if (!_objectViewSourceByBuffer.TryGetValue(current, out current!))
                {
                    yield break;
                }
            }
        }

        private static Dimension[] GetShapeArgument(BaseExpr expr, string context)
        {
            expr = UnwrapInputBoxing(expr);
            if (expr is Shape shape)
            {
                return GetRankedShapeDimensions(shape, context);
            }

            throw new NotSupportedException($"{context} expects a ranked shape, got {expr.GetType().Name}.");
        }

        private TIR.Buffer GetAbiBuffer(BufferVar bufferVar, string context)
        {
            if (_abiBufferMemo.TryGetValue(bufferVar, out var buffer))
            {
                return buffer;
            }

            var tensorType = GetTensorType(bufferVar.CheckedType, context);
            var distributedType = bufferVar.CheckedType as DistributedType;
            buffer = T.AttachBuffer(bufferVar, tensorType, bufferVar.Location, 0, out _, $"{bufferVar.Name}_abi", distributedType);
            _abiBufferMemo.Add(bufferVar, buffer);
            return buffer;
        }

        private void MarkStoredOutput(TIR.Buffer buffer, string context)
        {
            if (buffer.MemSpan.Buffer.Location != MemoryLocation.Output || IsFormalTensorBuffer(buffer))
            {
                return;
            }

            var outputIndex = GetOutputIndex(buffer);
            RecordTensorOutputStore(outputIndex, GetDistributedType(buffer), context);
        }

        private void RecordTensorOutputStore(
            int outputIndex,
            DistributedType? distributedType,
            string context)
        {
            if (_outputAliases.ContainsKey(outputIndex) || _formalObjectOutputAliases.ContainsKey(outputIndex))
            {
                throw new NotSupportedException(
                    $"PyNTT PrimFunction {_function.Name} materializes output {_outputs[outputIndex].Name} " +
                    $"both by an alias and {context}.");
            }

            var previousType = _outputDistributedTypes[outputIndex];
            if (previousType is not null &&
                distributedType is not null &&
                !previousType.Equals(distributedType))
            {
                throw new NotSupportedException(
                    $"PyNTT PrimFunction {_function.Name} stores output {_outputs[outputIndex].Name} " +
                    $"with incompatible distributed types: {previousType} versus {distributedType}.");
            }

            _storedOutputIndices.Add(outputIndex);
            _definitelyStoredOutputIndices.Add(outputIndex);
            _outputDistributedTypes[outputIndex] = previousType ?? distributedType;
        }

        private bool TryRecordStoredOutput(int outputIndex)
        {
            if (!_storedOutputIndices.Add(outputIndex))
            {
                return false;
            }

            _definitelyStoredOutputIndices.Add(outputIndex);
            return true;
        }

        private OutputControlFlowState CaptureOutputControlFlowState()
            => new(
                new HashSet<int>(_storedOutputIndices),
                new HashSet<int>(_definitelyStoredOutputIndices),
                _outputDistributedTypes.ToArray(),
                new Dictionary<int, int>(_outputAliases),
                new Dictionary<int, string>(_formalObjectOutputAliases));

        private void RestoreOutputControlFlowState(OutputControlFlowState state)
        {
            ReplaceSet(_storedOutputIndices, state.MayStore);
            ReplaceSet(_definitelyStoredOutputIndices, state.MustStore);
            Array.Copy(state.OutputDistributedTypes, _outputDistributedTypes, _outputDistributedTypes.Length);
            ReplaceDictionary(_outputAliases, state.OutputAliases);
            ReplaceDictionary(_formalObjectOutputAliases, state.FormalObjectOutputAliases);
        }

        private void MergeConditionalOutputStates(OutputControlFlowState thenState, OutputControlFlowState elseState)
        {
            if (!DictionaryEqual(thenState.OutputAliases, elseState.OutputAliases))
            {
                throw new NotSupportedException(
                    $"PyNTT PrimFunction {_function.Name} assigns runtime output aliases differently across conditional branches. " +
                    "Branch-dependent aliases cannot be represented by the function ABI.");
            }

            if (!DictionaryEqual(thenState.FormalObjectOutputAliases, elseState.FormalObjectOutputAliases))
            {
                throw new NotSupportedException(
                    $"PyNTT PrimFunction {_function.Name} assigns formal object output aliases differently across conditional branches. " +
                    "Branch-dependent aliases cannot be represented by the function ABI.");
            }

            var mergedMayStore = thenState.MayStore.Union(elseState.MayStore).ToHashSet();
            var mergedMustStore = thenState.MustStore.Intersect(elseState.MustStore).ToHashSet();
            var mergedDistributedTypes = new DistributedType?[_outputDistributedTypes.Length];
            for (var outputIndex = 0; outputIndex < mergedDistributedTypes.Length; outputIndex++)
            {
                var thenType = thenState.OutputDistributedTypes[outputIndex];
                var elseType = elseState.OutputDistributedTypes[outputIndex];
                if (thenState.MayStore.Contains(outputIndex) &&
                    elseState.MayStore.Contains(outputIndex) &&
                    thenType is not null &&
                    elseType is not null &&
                    !thenType.Equals(elseType))
                {
                    throw new NotSupportedException(
                        $"PyNTT PrimFunction {_function.Name} stores output {_outputs[outputIndex].Name} " +
                        $"with incompatible distributed types across conditional branches: {thenType} versus {elseType}.");
                }

                mergedDistributedTypes[outputIndex] = thenType ?? elseType;
            }

            RestoreOutputControlFlowState(new(
                mergedMayStore,
                mergedMustStore,
                mergedDistributedTypes,
                thenState.OutputAliases,
                thenState.FormalObjectOutputAliases));
        }

        private static void ReplaceSet<T>(HashSet<T> destination, IReadOnlySet<T> source)
        {
            destination.Clear();
            destination.UnionWith(source);
        }

        private static void ReplaceDictionary<TKey, TValue>(Dictionary<TKey, TValue> destination, IReadOnlyDictionary<TKey, TValue> source)
            where TKey : notnull
        {
            destination.Clear();
            foreach (var pair in source)
            {
                destination.Add(pair.Key, pair.Value);
            }
        }

        private static bool DictionaryEqual<TKey, TValue>(IReadOnlyDictionary<TKey, TValue> lhs, IReadOnlyDictionary<TKey, TValue> rhs)
            where TKey : notnull
            => lhs.Count == rhs.Count && lhs.All(pair => rhs.TryGetValue(pair.Key, out var value) && EqualityComparer<TValue>.Default.Equals(pair.Value, value));

        private string GetInputName(int inputIndex, string context)
        {
            if (inputIndex < 0 || inputIndex >= _inputNames.Count)
            {
                throw new NotSupportedException($"{context} references input index {inputIndex}, but PyNTT PrimFunction {_function.Name} only has {_inputNames.Count} inputs.");
            }

            return _inputNames[inputIndex];
        }

        private int GetOutputIndex(BaseExpr expr)
        {
            expr = ResolveBoundExpression(expr);
            if (TryGetDirectOutputName(expr, out var directOutputName) && TryFindOutputIndex(directOutputName, out var directOutputIndex))
            {
                return directOutputIndex;
            }

            var outputName = GetTensorName(expr, _parameterNames);
            if (TryFindOutputIndex(outputName, out var outputIndex))
            {
                return outputIndex;
            }

            throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} references unknown output parameter {outputName}.");
        }

        private bool TryGetOutputIndex(BaseExpr expr, out int outputIndex)
        {
            outputIndex = -1;
            if (TryGetDirectOutputName(expr, out var directOutputName))
            {
                return TryFindOutputIndex(directOutputName, out outputIndex);
            }

            if (!TryGetAbiTensorName(expr, requireLocation: MemoryLocation.Output, out var outputName))
            {
                return false;
            }

            return TryFindOutputIndex(outputName, out outputIndex);
        }

        private bool TryGetDirectOutputName(BaseExpr expr, out string outputName)
        {
            expr = ResolveBoundExpression(expr);
            outputName = string.Empty;
            if (expr is BufferVar { Role: BufferVarRole.Output } outputVar)
            {
                outputName = outputVar.Name;
                return true;
            }

            if (expr is TIR.Buffer buffer &&
                IsAbiBuffer(buffer) &&
                buffer.MemSpan.Buffer.Location == MemoryLocation.Output &&
                buffer.MemSpan.Buffer.Start is BufferVar { Role: BufferVarRole.Output } outputBufferVar)
            {
                outputName = outputBufferVar.Name;
                return true;
            }

            return false;
        }

        private bool TryFindOutputIndex(string outputName, out int outputIndex)
        {
            for (var i = 0; i < _outputs.Length; i++)
            {
                if (_outputs[i].Name == outputName || _outputs[i].AbiName == outputName)
                {
                    outputIndex = i;
                    return true;
                }
            }

            outputIndex = -1;
            return false;
        }

        private bool TryGetDirectInputName(BaseExpr expr, out string inputName)
        {
            if (TryGetFormalObjectBaseName(expr, out _))
            {
                inputName = string.Empty;
                return false;
            }

            if (TryGetAbiTensorName(expr, requireLocation: MemoryLocation.Input, out inputName))
            {
                return true;
            }

            inputName = string.Empty;
            return false;
        }

        private bool TryGetAbiTensorName(BaseExpr expr, MemoryLocation requireLocation, out string name)
        {
            expr = ResolveBoundExpression(expr);
            name = string.Empty;
            if (expr is TIR.Buffer buffer)
            {
                if (!IsAbiBuffer(buffer) ||
                    buffer.MemSpan.Buffer.Location != requireLocation ||
                    buffer.MemSpan.Buffer.Start is not IVar bufferParameter ||
                    !_parameterNames.TryGetValue(bufferParameter, out var bufferParameterName))
                {
                    return false;
                }

                name = bufferParameterName;
                return true;
            }

            if (expr is BufferVar bufferVar)
            {
                if (bufferVar.Location != requireLocation || !_parameterNames.TryGetValue(bufferVar, out var bufferVarName))
                {
                    return false;
                }

                name = bufferVarName;
                return true;
            }

            if (requireLocation == MemoryLocation.Input &&
                expr is IVar parameter &&
                _parameterNames.TryGetValue(parameter, out var parameterName) &&
                !_outputs.Any(output => output.Name == parameterName || output.AbiName == parameterName))
            {
                name = parameterName;
                return true;
            }

            return false;
        }

        private int GetBufferInputIndex(TIR.Buffer buffer, string context)
        {
            foreach (var aliasBuffer in EnumerateObjectAliasBuffers(buffer))
            {
                if (_bufferInputIndices.TryGetValue(aliasBuffer, out var inputIndex))
                {
                    return inputIndex;
                }
            }

            throw new NotSupportedException($"{context} must be loaded from a function input before use.");
        }

        private string GetKVCacheFieldArgument(BaseExpr expr, string field, PyNTTKVCacheStorageMetadata? storage = null)
        {
            expr = ResolveBoundExpression(expr);
            if (TryGetFormalObjectBaseName(expr, out var objectBaseName))
            {
                var argumentName = SanitizePythonIdentifier($"{objectBaseName}_{field}");
                _extraWorkspaceBaseNames.Add(argumentName);
                var tritonDType = field switch
                {
                    "kv_caches" => storage is null
                        ? throw new NotSupportedException($"PyNTT formal KV-cache field {argumentName} is missing storage metadata.")
                        : GetTritonDType(storage.DType),
                    "metadata" or "slot_mapping" or "block_tables" => "tl.int64",
                    "kv_caches_blocks" => null,
                    _ => throw new NotSupportedException($"PyNTT formal KV-cache field {argumentName} has unknown ABI kind."),
                };
                if (tritonDType is not null &&
                    _extraPointerParameterTritonTypes.TryGetValue(argumentName, out var existingTritonDType) &&
                    existingTritonDType != tritonDType)
                {
                    throw new NotSupportedException($"PyNTT formal KV-cache field {argumentName} has conflicting pointer types {existingTritonDType} and {tritonDType}.");
                }

                if (tritonDType is not null)
                {
                    _extraPointerParameterTritonTypes[argumentName] = tritonDType;
                }

                _formalObjectFieldStorages[argumentName] = storage;
                return argumentName;
            }

            return $"input{RegisterKVCacheFieldInput(expr, field, storage).ToString(CultureInfo.InvariantCulture)}";
        }

        private int RegisterKVCacheFieldInput(BaseExpr expr, string field, PyNTTKVCacheStorageMetadata? storage = null)
        {
            var sourceName = GetKVCacheSourceInputName(expr, $"PyNTT KV-cache field {field}");
            var syntheticName = $"{sourceName}.__{field}";
            var inputIndex = _inputNames.IndexOf(syntheticName);
            if (inputIndex < 0)
            {
                inputIndex = _inputNames.Count;
                _inputNames.Add(syntheticName);
            }

            if (!_kvCacheFieldInputs.Any(input => input.Name == syntheticName))
            {
                _kvCacheFieldInputs.Add(new(syntheticName, sourceName, field, storage));
            }

            return inputIndex;
        }

        private bool TryGetFormalObjectBaseName(BaseExpr expr, out string baseName)
        {
            expr = ResolveBoundExpression(expr);
            baseName = string.Empty;
            if (expr is IVar parameter && _formalObjectParameterBaseNames.TryGetValue(parameter, out var parameterBaseName))
            {
                baseName = parameterBaseName;
                return true;
            }

            if (expr is TIR.Buffer buffer)
            {
                foreach (var aliasBuffer in EnumerateObjectAliasBuffers(buffer))
                {
                    if (_formalObjectBaseNameByBuffer.TryGetValue(aliasBuffer, out var aliasBaseName))
                    {
                        baseName = aliasBaseName;
                        return true;
                    }

                    if (IsAbiBuffer(aliasBuffer) &&
                        aliasBuffer.MemSpan.Buffer.Start is IVar bufferParameter &&
                        _formalObjectParameterBaseNames.TryGetValue(bufferParameter, out var bufferBaseName))
                    {
                        baseName = bufferBaseName;
                        return true;
                    }
                }
            }

            return false;
        }

        private string GetKVCacheSourceInputName(BaseExpr expr, string context)
        {
            expr = UnwrapInputBoxing(expr);
            if (expr is TIR.Buffer kvCache)
            {
                if (IsAbiBuffer(kvCache))
                {
                    return _inputNames[GetInputIndex(kvCache)];
                }

                return _inputNames[GetBufferInputIndex(kvCache, context)];
            }

            return GetTensorName(expr, _parameterNames);
        }

        private KernelInputLayout BuildKernelInputLayout(string bodySource, IReadOnlyList<DeviceFunctionRenderSpec> deviceFunctions)
        {
            var names = new List<string>(_inputNames.Count);
            var indexMap = new Dictionary<int, int>();
            var removedIndexes = new HashSet<int>();

            for (var i = 0; i < _inputNames.Count; i++)
            {
                var name = _inputNames[i];
                if (IsObjectKernelInput(name))
                {
                    removedIndexes.Add(i);
                    continue;
                }

                indexMap[i] = names.Count;
                names.Add(name);
            }

            var remappedBodySource = RemapInputReferences(bodySource, indexMap, removedIndexes, $"PyNTT PrimFunction {_function.Name} body");
            var helpers = _helpers
                .Select((helper, index) => RemapHelperInputReferences(helper, indexMap, removedIndexes, $"PyNTT PrimFunction {_function.Name} helper {index}"))
                .ToArray();
            var remappedDeviceFunctions = deviceFunctions
                .Select(deviceFunction => RemapDeviceFunctionInputReferences(deviceFunction, indexMap, removedIndexes))
                .ToArray();
            return new(names.ToArray(), indexMap, removedIndexes, remappedBodySource, helpers, remappedDeviceFunctions);
        }

        private bool IsObjectKernelInput(string name)
        {
            foreach (var pair in _parameterNames)
            {
                if (pair.Value == name && pair.Key is Expr expr)
                {
                    return IsObjectExpression(expr);
                }
            }

            return false;
        }

        private static HelperTemplateRenderSpec RemapHelperInputReferences(
            HelperTemplateRenderSpec helper,
            IReadOnlyDictionary<int, int> indexMap,
            IReadOnlySet<int> removedIndexes,
            string context)
        {
            var model = JsonSerializer.SerializeToNode(helper.Model);
            var remappedModel = RemapJsonInputReferences(model, indexMap, removedIndexes, context);
            var remappedArguments = helper.Arguments
                .Select(argument => RemapInputReferences(argument, indexMap, removedIndexes, context))
                .ToArray();
            return helper with { Model = remappedModel ?? new JsonObject(), Arguments = remappedArguments };
        }

        private DeviceFunctionRenderSpec RemapDeviceFunctionInputReferences(
            DeviceFunctionRenderSpec deviceFunction,
            IReadOnlyDictionary<int, int> indexMap,
            IReadOnlySet<int> removedIndexes)
        {
            var context = $"PyNTT device function {deviceFunction.Name}";
            var helpers = deviceFunction.Helpers
                .Select((helper, index) => RemapHelperInputReferences(helper, indexMap, removedIndexes, $"{context} helper {index}"))
                .ToArray();
            var parameterOverrides = deviceFunction.ParameterOverrides.ToDictionary(
                pair => pair.Key,
                pair => RemapInputReferences(pair.Value, indexMap, removedIndexes, $"{context} parameter override {pair.Key}"),
                StringComparer.Ordinal);
            var extraParameterArguments = deviceFunction.ExtraParameterArguments.ToDictionary(
                pair => pair.Key,
                pair => RemapInputReferences(pair.Value, indexMap, removedIndexes, $"{context} extra parameter {pair.Key}"),
                StringComparer.Ordinal);
            return deviceFunction with
            {
                Helpers = helpers,
                BodySource = RemapInputReferences(deviceFunction.BodySource, indexMap, removedIndexes, $"{context} body"),
                ParameterOverrides = parameterOverrides,
                ExtraParameterArguments = extraParameterArguments,
            };
        }

        private static JsonNode? RemapJsonInputReferences(
            JsonNode? node,
            IReadOnlyDictionary<int, int> indexMap,
            IReadOnlySet<int> removedIndexes,
            string context)
        {
            switch (node)
            {
                case null:
                    return null;
                case JsonValue value when value.TryGetValue<string>(out var text):
                    return JsonValue.Create(RemapInputReferences(text, indexMap, removedIndexes, context));
                case JsonValue value:
                    return value.DeepClone();
                case JsonArray array:
                    var remappedArray = new JsonArray();
                    foreach (var item in array)
                    {
                        remappedArray.Add(RemapJsonInputReferences(item, indexMap, removedIndexes, context));
                    }

                    return remappedArray;
                case JsonObject obj:
                    var remappedObject = new JsonObject();
                    foreach (var pair in obj)
                    {
                        remappedObject[pair.Key] = RemapJsonInputReferences(pair.Value, indexMap, removedIndexes, context);
                    }

                    return remappedObject;
                default:
                    return node.DeepClone();
            }
        }

        private static string RemapInputReferences(
            string source,
            IReadOnlyDictionary<int, int> indexMap,
            IReadOnlySet<int> removedIndexes,
            string context)
        {
            return InputReferenceRegex.Replace(source, match =>
            {
                var inputIndex = int.Parse(match.Groups["index"].Value, CultureInfo.InvariantCulture);
                if (removedIndexes.Contains(inputIndex))
                {
                    throw new NotSupportedException($"{context} references object input{inputIndex}, which cannot be passed to a Triton kernel.");
                }

                return indexMap.TryGetValue(inputIndex, out var remappedInputIndex)
                    ? $"input{remappedInputIndex.ToString(CultureInfo.InvariantCulture)}{match.Groups["suffix"].Value}"
                    : match.Value;
            });
        }

        private PyNTTPagedAttentionCacheTemplateModel GetPagedAttentionCacheTemplateModel(BaseExpr expr, string context)
        {
            var config = GetPagedAttentionConfig(expr, context);
            ValidatePagedAttentionConfig(config, context);

            var key = GetPagedAttentionCacheKindLayout(config, AttentionCacheKind.Key, context);
            var value = GetPagedAttentionCacheKindLayout(config, AttentionCacheKind.Value, context);
            var topologyShape = GetPagedAttentionTopologyShape(config);
            var numBlocksSplitAxes = GetPagedAttentionNumBlocksSplitAxes(config);
            var valueSectionOffset = key.SectionElements;
            var blockElements = checked(key.SectionElements + value.SectionElements);
            return new(
                GetPyNTTScalarDTypeName(config.KVPrimType),
                GetScalarTritonDType(config.KVPrimType),
                config.NumLayers,
                config.NumKVHeads,
                config.HeadDim,
                config.BlockSize,
                key.LaneCount,
                value.LaneCount,
                (int)key.VectorizedDim,
                (int)value.VectorizedDim,
                key.HeadDimBlocks,
                value.HeadDimBlocks,
                0,
                valueSectionOffset,
                key.SectionElements,
                value.SectionElements,
                blockElements,
                key.LayerStride,
                key.HeadStride,
                key.DimBlockStride,
                key.BlockOffsetStride,
                value.LayerStride,
                value.HeadStride,
                value.DimBlockStride,
                value.BlockOffsetStride,
                key.TailShape,
                value.TailShape,
                config.ShardingAxes.Count + 1,
                topologyShape,
                numBlocksSplitAxes);
        }

        private static IPagedAttentionConfig GetPagedAttentionConfig(BaseExpr expr, string context)
        {
            if (expr is TIR.Buffer buffer)
            {
                return GetPagedAttentionConfig(buffer.ElemType, context);
            }

            var tensorType = GetTensorType(expr.CheckedType, context);
            return GetPagedAttentionConfig(tensorType.DType, context);
        }

        private static IPagedAttentionConfig GetPagedAttentionConfig(DataType dataType, string context)
        {
            if (dataType is ReferenceType { ElemType: PagedAttentionKVCacheType kvCacheType } &&
                kvCacheType.Config is { } config)
            {
                return config;
            }

            throw new NotSupportedException($"{context} expects Reference<PagedAttentionKVCacheType>, got {dataType}.");
        }

        private sealed record PagedAttentionCacheKindLayout(
            int LaneCount,
            PagedKVCacheDimKind VectorizedDim,
            int HeadDimBlocks,
            int[] TailShape,
            int SectionElements,
            int LayerStride,
            int HeadStride,
            int DimBlockStride,
            int BlockOffsetStride);

        private static PagedAttentionCacheKindLayout GetPagedAttentionCacheKindLayout(IPagedAttentionConfig config, AttentionCacheKind kind, string context)
        {
            var vectorizedAxes = config.GetVectorizedAxes(kind).ToArray();
            var lanes = config.GetLanes(kind).ToArray();
            int laneCount;
            PagedKVCacheDimKind vectorizedDim;
            if (vectorizedAxes.Length == 0 && lanes.Length == 0)
            {
                laneCount = 1;
                vectorizedDim = (PagedKVCacheDimKind)(-1);
            }
            else if (vectorizedAxes.Length == 1 && lanes.Length == 1 &&
                vectorizedAxes[0] is PagedKVCacheDimKind.HeadDim or PagedKVCacheDimKind.BlockSize)
            {
                laneCount = checked((int)lanes[0]);
                vectorizedDim = vectorizedAxes[0];
            }
            else
            {
                throw new NotSupportedException($"{context} currently supports only {kind} HeadDim or BlockSize vectorization with zero or one lane value.");
            }

            if (laneCount <= 0)
            {
                throw new NotSupportedException($"{context} requires {kind} lane to be positive, got {laneCount}.");
            }

            if (vectorizedDim == PagedKVCacheDimKind.HeadDim && config.HeadDim % laneCount != 0)
            {
                throw new NotSupportedException($"{context} requires {kind} head_dim divisible by lane, got head_dim={config.HeadDim}, lane={laneCount}.");
            }

            if (vectorizedDim == PagedKVCacheDimKind.BlockSize && config.BlockSize % laneCount != 0)
            {
                throw new NotSupportedException($"{context} requires {kind} block_size divisible by lane, got block_size={config.BlockSize}, lane={laneCount}.");
            }

            var headDimBlocks = vectorizedDim == PagedKVCacheDimKind.HeadDim
                ? checked(config.HeadDim / laneCount)
                : config.HeadDim;
            var tailDims = new List<int>();
            var tailDimKinds = new List<PagedKVCacheDimKind>();
            foreach (var dimKind in config.GetCacheLayout(kind))
            {
                if (dimKind is PagedKVCacheDimKind.NumBlocks or PagedKVCacheDimKind.KV)
                {
                    continue;
                }

                tailDimKinds.Add(dimKind);
                var dimExtent = dimKind switch
                {
                    PagedKVCacheDimKind.NumLayers => checked((int)config.NumLayers),
                    PagedKVCacheDimKind.BlockSize => checked((int)config.BlockSize),
                    PagedKVCacheDimKind.NumKVHeads => checked((int)config.NumKVHeads),
                    PagedKVCacheDimKind.HeadDim => checked((int)config.HeadDim),
                    _ => throw new NotSupportedException($"{context} does not support {kind} cache dimension {dimKind}."),
                };
                if (dimKind == vectorizedDim)
                {
                    dimExtent = checked(dimExtent / laneCount);
                }

                tailDims.Add(dimExtent);
            }

            if (tailDims.Count != 4)
            {
                throw new NotSupportedException($"{context} expects {kind} cache layout to contain exactly NumLayers, NumKVHeads, HeadDim, and BlockSize after removing NumBlocks/KV.");
            }

            var strides = ComputeContiguousStrides(tailDims);
            int GetStride(PagedKVCacheDimKind dimKind)
            {
                var index = tailDimKinds.IndexOf(dimKind);
                if (index < 0)
                {
                    throw new NotSupportedException($"{context} {kind} cache layout is missing {dimKind}.");
                }

                return strides[index];
            }

            var sectionVectorElements = checked(tailDims.Aggregate(1, static (acc, dim) => checked(acc * dim)));
            var sectionElements = checked(sectionVectorElements * laneCount);
            return new(
                laneCount,
                vectorizedDim,
                headDimBlocks,
                tailDims.ToArray(),
                sectionElements,
                GetStride(PagedKVCacheDimKind.NumLayers),
                GetStride(PagedKVCacheDimKind.NumKVHeads),
                GetStride(PagedKVCacheDimKind.HeadDim),
                GetStride(PagedKVCacheDimKind.BlockSize));
        }

        private static int[] ComputeContiguousStrides(IReadOnlyList<int> shape)
        {
            var strides = new int[shape.Count];
            var stride = 1;
            for (int i = shape.Count - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride = checked(stride * shape[i]);
            }

            return strides;
        }

        private PyNTTKVCacheStorageMetadata GetKVCacheStorageMetadata(PyNTTPagedAttentionCacheTemplateModel cache)
        {
            return new(
                cache.DType,
                cache.TopologyShape,
                cache.KeyTailShape,
                cache.ValueTailShape,
                cache.KeySectionElements,
                cache.ValueSectionElements,
                cache.BlockElements,
                cache.BlockSize);
        }

        private static int GetGlobalNumQueryHeads(Nncase.TIR.NTT.PagedAttention pagedAttention, PyNTTPagedAttentionCacheTemplateModel cache)
        {
            if (pagedAttention.HiddenSize <= 0 || pagedAttention.HiddenSize % cache.HeadDim != 0)
            {
                throw new NotSupportedException($"PyNTT PagedAttention requires hidden_size divisible by head_dim, got hidden_size={pagedAttention.HiddenSize}, head_dim={cache.HeadDim}.");
            }

            var numQueryHeads = checked(pagedAttention.HiddenSize / cache.HeadDim);
            if (numQueryHeads <= 0 || numQueryHeads % cache.NumKVHeads != 0)
            {
                throw new NotSupportedException($"PyNTT PagedAttention requires num_query_heads divisible by num_kv_heads, got num_query_heads={numQueryHeads}, num_kv_heads={cache.NumKVHeads}.");
            }

            return numQueryHeads;
        }

        private void ValidatePagedAttentionConfig(IPagedAttentionConfig config, string context)
        {
            ValidatePagedAttentionLayout(config.KeyCacheLayout, AttentionCacheKind.Key, context);
            ValidatePagedAttentionLayout(config.ValueCacheLayout, AttentionCacheKind.Value, context);
            _ = GetPagedAttentionCacheKindLayout(config, AttentionCacheKind.Key, context);
            _ = GetPagedAttentionCacheKindLayout(config, AttentionCacheKind.Value, context);

            if (config.ShardingAxes.Count > 1 ||
                (config.ShardingAxes.Count == 1 && config.ShardingAxes[0] != PagedKVCacheDimKind.NumBlocks))
            {
                throw new NotSupportedException($"{context} currently supports no KV-cache sharding or NumBlocks-only sharding.");
            }

            if (config.AxisPolicies.Count != config.ShardingAxes.Count)
            {
                throw new NotSupportedException($"{context} requires one axis policy per KV-cache sharding axis.");
            }

            if (config.ShardingAxes.Count == 1)
            {
                var hierarchy = GetBlockHierarchy(_targetOptions);
                var policy = config.AxisPolicies[0];
                foreach (var axis in policy.Axes)
                {
                    if (axis < 0 || axis >= hierarchy.Length)
                    {
                        throw new NotSupportedException($"{context} NumBlocks sharding axis {axis} is outside PyNTT hierarchy rank {hierarchy.Length}.");
                    }
                }
            }
        }

        private static void ValidatePagedAttentionLayout(IRArray<PagedKVCacheDimKind> layout, AttentionCacheKind kind, string context)
        {
            if (layout.Count != 6)
            {
                throw new NotSupportedException($"{context} {kind} cache layout must have 6 dimensions, got {layout.Count}.");
            }

            var expected = Enum.GetValues<PagedKVCacheDimKind>().OrderBy(x => (int)x).ToArray();
            var actual = layout.ToArray().OrderBy(x => (int)x).ToArray();
            if (!actual.SequenceEqual(expected))
            {
                throw new NotSupportedException($"{context} {kind} cache layout must be a permutation of [{string.Join(", ", expected)}], got [{string.Join(", ", layout)}].");
            }
        }

        private int[] GetPagedAttentionTopologyShape(IPagedAttentionConfig config)
        {
            if (config.ShardingAxes.Count == 0)
            {
                return Array.Empty<int>();
            }

            var hierarchy = GetBlockHierarchy(_targetOptions);
            return config.AxisPolicies
                .Select(policy => policy.Axes.Aggregate(1, (product, axis) => checked(product * hierarchy[axis])))
                .ToArray();
        }

        private static int[] GetPagedAttentionNumBlocksSplitAxes(IPagedAttentionConfig config)
        {
            if (config.ShardingAxes.Count == 0)
            {
                return Array.Empty<int>();
            }

            return config.AxisPolicies[0].Axes.ToArray();
        }

        private PyNTTBufferPointerTemplateModel GetBufferPointer(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer);
            return new(
                BuildPointerExpression(bufferRef, GetTritonDType(buffer.ElemType)),
                bufferRef.ShardCoordHierarchy,
                bufferRef.AddressSpace,
                TryCreateLocalBuffer(buffer, bufferRef));
        }

        private PyNTTBufferPointerTemplateModel GetBufferScalarPointer(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer);
            return new(
                BuildPointerExpression(bufferRef, GetScalarTritonDType(buffer.ElemType)),
                bufferRef.ShardCoordHierarchy,
                bufferRef.AddressSpace,
                TryCreateLocalBuffer(buffer, bufferRef));
        }

        private PyNTTBufferPointerTemplateModel GetBufferScalarPointer(TIR.Buffer buffer, string indexExpression)
        {
            var bufferRef = ResolveBufferRef(buffer) with { IndexExpression = indexExpression };
            return new(
                BuildPointerExpression(bufferRef, GetScalarTritonDType(buffer.ElemType)),
                bufferRef.ShardCoordHierarchy,
                bufferRef.AddressSpace,
                TryCreateLocalBuffer(buffer, bufferRef));
        }

        private PyNTTBufferPointerTemplateModel GetRegionCopyBufferPointer(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer);
            var pointer = BuildPointerExpression(bufferRef, GetScalarTritonDType(buffer.ElemType));
            var localBuffer = TryCreateLocalBuffer(buffer, bufferRef);
            return new(pointer, bufferRef.ShardCoordHierarchy, bufferRef.AddressSpace, localBuffer);
        }

        private PyNTTLocalBufferTemplateModel? TryCreateLocalBuffer(
            TIR.Buffer buffer,
            BufferRef bufferRef)
        {
            if (bufferRef.SharedAllocation is not { } sharedAllocation ||
                bufferRef.PoolStrideBytes != "0" ||
                bufferRef.IndexExpression is not null ||
                bufferRef.IsByteAddressed ||
                sharedAllocation.ScalarElementSizeBytes != GetScalarElementSizeBytes(buffer.ElemType) ||
                sharedAllocation.TritonDType != GetScalarTritonDType(buffer.ElemType))
            {
                return null;
            }

            return new(
                sharedAllocation.DescriptorName,
                sharedAllocation.DescriptorShape,
                sharedAllocation.LogicalShape,
                sharedAllocation.LogicalStrides,
                bufferRef.LocalBaseCoordinates ?? throw new InvalidOperationException(
                    $"PyNTT shared buffer {buffer.Name} is missing logical base coordinates."),
                sharedAllocation.VectorLaneShape,
                sharedAllocation.AvailableBytes,
                sharedAllocation.ScalarElementSizeBytes,
                sharedAllocation.StorageEncoding);
        }

        private string BuildFormalTensorBasePointerArgument(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer) with { IndexExpression = null };
            return BuildPointerExpression(bufferRef, GetScalarTritonDType(buffer.ElemType));
        }

        private string BuildFormalTensorPoolStrideElementsArgument(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer);
            return bufferRef.IsByteAddressed
                ? DivideOffsetExpression(bufferRef.PoolStrideBytes, GetScalarElementSizeBytes(buffer.ElemType))
                : bufferRef.PoolStrideBytes;
        }

        private string BuildFormalTensorPoolScopeSizeArgument(TIR.Buffer buffer)
            => ResolveBufferRef(buffer).PoolScopeSize;

        private PyNTTDimExpression[] GetTensorShape(BaseExpr expr, string name)
        {
            var tensorType = GetTensorType(expr.CheckedType, name);
            return GetRankedShape(tensorType, name).Dimensions.ToArray()
                .Select(_dimEmitter.Emit)
                .ToArray();
        }

        private BufferRef ResolveBufferRef(TIR.Buffer buffer)
        {
            if (_sharedBufferAllocations.TryGetValue(buffer, out var sharedAllocation))
            {
                return new BufferRef(
                    sharedAllocation.DescriptorName,
                    "0",
                    "0",
                    null,
                    null,
                    false,
                    AddressSpace: 3,
                    SharedAllocation: sharedAllocation,
                    LocalBaseCoordinates: CreateZeroDimExpressions(sharedAllocation.LogicalShape.Length));
            }

            if (_bufferViewSourceByBuffer.TryGetValue(buffer, out var viewSource))
            {
                return ResolveBufferViewRef(viewSource);
            }

            if (TryGetFormalTensorBaseName(buffer, out var formalBaseName))
            {
                return ResolveFormalAbiBufferRef(buffer, formalBaseName);
            }

            if (buffer.MemSpan.Buffer.Location == MemoryLocation.Input)
            {
                return ResolveAbiBufferRef(buffer, $"input{GetInputIndex(buffer).ToString(CultureInfo.InvariantCulture)}");
            }

            if (buffer.MemSpan.Buffer.Location == MemoryLocation.Output)
            {
                return ResolveAbiBufferRef(buffer, $"output{GetOutputIndex(buffer).ToString(CultureInfo.InvariantCulture)}");
            }

            var offsetBytes = GetBufferOffsetBytes(buffer);
            var shardCoordHierarchy = RequiresShardCoords(offsetBytes)
                ? GetShardCoordHierarchy(buffer)
                : null;
            if (buffer.MemSpan.Buffer.Location == MemoryLocation.Shared)
            {
                throw new NotSupportedException(
                    $"PyNTT shared buffer {buffer.Name} is used outside its TIR AllocateBufferView scope. " +
                    "Shared storage must be allocated by a Let-bound AllocateBufferView before any TIR operation references it.");
            }

            return buffer.MemSpan.Buffer.Location switch
            {
                MemoryLocation.Data when buffer.DistributedType is null => new(GetDataBaseName(buffer), offsetBytes, "0", null, shardCoordHierarchy, true),
                MemoryLocation.Data => new(GetDataBaseName(buffer), offsetBytes, "data_pool_stride_bytes", "shard_index", shardCoordHierarchy, true),
                MemoryLocation.ChipLocalData => new(GetChipLocalDataBaseName(buffer), offsetBytes, "0", null, shardCoordHierarchy, true),
                MemoryLocation.BlockLocalData => CreateBlockLocalBufferRef(buffer, offsetBytes, shardCoordHierarchy),
                MemoryLocation.Rdata => new("rdata", offsetBytes, "0", null, shardCoordHierarchy, true),
                MemoryLocation.ChipLocalRdata => new("chip_local_rdata", offsetBytes, "0", null, shardCoordHierarchy, true),
                MemoryLocation.BlockLocalRdata => new("block_local_rdata", offsetBytes, PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_currentFunction.SchedResult.BlockLocalRdatas, _targetOptions, "b").ToString(CultureInfo.InvariantCulture), "shard_index", shardCoordHierarchy, true),
                var location => throw new NotSupportedException($"PyNTT does not support buffer memory location {location} for Triton template operands yet."),
            };
        }

        private BufferRef ResolveByteAddressedBufferRef(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer);
            if (bufferRef.SharedAllocation is { } sharedAllocation)
            {
                throw new NotSupportedException(
                    $"PyNTT shared descriptor {sharedAllocation.DescriptorName} cannot be converted to a byte address. " +
                    "The consuming helper must use PyNTTBufferPointerTemplateModel.LocalBuffer and tle.gpu.local_ptr.");
            }

            if (bufferRef.IsByteAddressed)
            {
                return bufferRef;
            }

            var scalarElementSizeBytes = GetScalarElementSizeBytes(buffer.ElemType);
            return bufferRef with
            {
                BaseName = $"({bufferRef.BaseName}).to({GetPointerTypeExpression("tl.uint8", bufferRef.AddressSpace)})",
                OffsetBytes = MultiplyOffsetExpression(bufferRef.OffsetBytes, scalarElementSizeBytes),
                PoolStrideBytes = MultiplyOffsetExpression(bufferRef.PoolStrideBytes, scalarElementSizeBytes),
                IsByteAddressed = true,
            };
        }

        private static PyNTTPooledByteAddressTemplateModel GetPooledByteAddressTemplateModel(BufferRef bufferRef)
            => new(
                bufferRef.BaseName,
                bufferRef.OffsetBytes,
                bufferRef.PoolStrideBytes,
                bufferRef.PoolScopeSize,
                bufferRef.AddressSpace);

        private BufferRef CreateBlockLocalBufferRef(TIR.Buffer buffer, string offsetBytes, int[]? shardCoordHierarchy)
        {
            var poolScopeSize = GetBlockLocalDataScopeSize(_targetOptions);
            return new(
                GetBlockLocalDataBaseName(buffer),
                offsetBytes,
                "block_local_data_pool_stride_bytes",
                BuildScopeIndexExpression("shard_index", poolScopeSize),
                shardCoordHierarchy,
                true,
                poolScopeSize.ToString(CultureInfo.InvariantCulture));
        }

        private BufferRef ResolveBufferViewRef(BufferViewSource viewSource)
        {
            var source = viewSource.Source;
            var sourceRef = ResolveBufferRef(source);
            if (sourceRef.SharedAllocation is { } sharedAllocation)
            {
                var sourceCoordinates = sourceRef.LocalBaseCoordinates
                    ?? throw new InvalidOperationException(
                        $"PyNTT shared buffer view {source.Name} is missing logical base coordinates.");
                if (sourceCoordinates.Length != viewSource.Offsets.Length ||
                    sourceCoordinates.Length != sharedAllocation.LogicalShape.Length)
                {
                    throw new NotSupportedException(
                        $"PyNTT shared buffer view rank mismatch: allocation logical rank={sharedAllocation.LogicalShape.Length}, " +
                        $"source coordinates={sourceCoordinates.Length}, offsets={viewSource.Offsets.Length}.");
                }

                var baseCoordinates = sourceCoordinates
                    .Zip(viewSource.Offsets)
                    .Select(pair => AddDimExpression(pair.First, pair.Second))
                    .ToArray();
                var sharedShardCoordHierarchy = baseCoordinates.Any(coordinate => RequiresShardCoords(coordinate.TritonExpression))
                    ? sourceRef.ShardCoordHierarchy ?? GetShardCoordHierarchy(source)
                    : sourceRef.ShardCoordHierarchy;
                return sourceRef with
                {
                    ShardCoordHierarchy = sharedShardCoordHierarchy,
                    LocalBaseCoordinates = baseCoordinates,
                };
            }

            var sourceStrides = GetBufferStrides(source);
            if (sourceStrides.Length != viewSource.Offsets.Length)
            {
                throw new NotSupportedException($"PyNTT buffer view rank mismatch: source strides={sourceStrides.Length}, offsets={viewSource.Offsets.Length}.");
            }

            var vectorElementOffset = PyNTTDimExpression.Zero;
            for (var axis = 0; axis < sourceStrides.Length; axis++)
            {
                vectorElementOffset = AddDimExpression(
                    vectorElementOffset,
                    MultiplyDimExpressions(sourceStrides[axis], viewSource.Offsets[axis]));
            }

            var scalarElementOffset = MultiplyDim(vectorElementOffset, GetVectorLaneElementCount(source.ElemType));
            var storageOffset = sourceRef.IsByteAddressed
                ? MultiplyDim(scalarElementOffset, GetScalarElementSizeBytes(source.ElemType))
                : scalarElementOffset;
            var combinedOffset = AddOffsetExpressions(sourceRef.OffsetBytes, storageOffset.TritonExpression);
            var shardCoordHierarchy = RequiresShardCoords(combinedOffset)
                ? sourceRef.ShardCoordHierarchy ?? GetShardCoordHierarchy(source)
                : null;
            return sourceRef with
            {
                OffsetBytes = combinedOffset,
                ShardCoordHierarchy = shardCoordHierarchy,
            };
        }

        private long GetBlockLocalDataPoolBytes()
        {
            var scheduledPoolBytes = checked((long)_currentFunction.SchedResult.BlockLocalDataPoolSize);
            return Math.Max(scheduledPoolBytes, _nestedBlockLocalDataPoolBytes);
        }

        private long GetSharedMemoryBytes()
        {
            var localBytes = GetMemoryLocationPoolSizeBytes(_bodyExpr, MemoryLocation.Shared);
            var requiredBytes = Math.Max(Math.Max(localBytes, _nestedSharedMemoryBytes), _sharedAliasRequiredBytes);
            if (requiredBytes == 0)
            {
                return 0;
            }

            var sharedSpace = _targetOptions.TargetMachineModel.TilingMemorySpaces
                .SingleOrDefault(space => space.TIRBinding?.Location == MemoryLocation.Shared)
                ?? throw new InvalidOperationException($"PyNTT function {_currentFunction.Name} uses Shared buffers, but target machine {_targetOptions.TargetMachineModel.Id} does not expose a Shared tiling memory space.");
            var allocationBytes = _targetOptions.TargetMachineModel.GetAllocationSizeBytes(sharedSpace, requiredBytes);
            if (allocationBytes > sharedSpace.MaxAllocationBytesPerScope)
            {
                throw new InvalidOperationException($"PyNTT function {_currentFunction.Name} requires {requiredBytes} shared-memory bytes ({allocationBytes} bytes after {sharedSpace.AllocationSizePolicy} allocation rounding), exceeding target machine {_targetOptions.TargetMachineModel.Id} allocation limit {sharedSpace.MaxAllocationBytesPerScope}.");
            }

            return allocationBytes;
        }

        private void AddTargetResourceMetadata()
        {
            var machine = _targetOptions.TargetMachineModel;
            var sharedSpace = machine.MemorySpaces.Values.SingleOrDefault(space => machine.GetMemoryResource(space).Kind == TargetMemorySpaceKind.Shared)
                ?? throw new InvalidOperationException($"PyNTT target machine {machine.Id} does not define a shared memory space.");
            var sharedResource = machine.GetMemoryResource(sharedSpace);
            _attrs["target_machine"] = machine.Id;
            _attrs["block_microkernels"] = _blockMicroKernelContracts;
            _attrs["target_private_resources"] = machine.PrivateResources.Values
                .OrderBy(resource => resource.Id.Value, StringComparer.Ordinal)
                .ToDictionary(
                    resource => resource.Id.Value,
                    resource => (object)new Dictionary<string, object>(StringComparer.Ordinal)
                    {
                        ["unit"] = resource.Unit.ToString(),
                        ["capacity_units"] = resource.CapacityUnits,
                        ["allocation_granularity_units"] = resource.AllocationGranularityUnits,
                        ["backing_memory_resource"] = resource.BackingMemoryResource?.Value ?? string.Empty,
                    },
                    StringComparer.Ordinal);
            if (machine.PrivateResources.TryGetValue(NTTTargetMachineCatalog.GpuRegisterFile, out var registerFile))
            {
                if (registerFile.Unit != TargetPrivateResourceUnit.Register32 ||
                    registerFile.CapacityUnits % machine.Execution.ThreadsPerBlock != 0)
                {
                    throw new InvalidOperationException(
                        $"PyNTT target {machine.Id} has an invalid aggregate GPU register-file specification.");
                }

                _attrs["registers_per_thread_limit"] = registerFile.CapacityUnits / machine.Execution.ThreadsPerBlock;
            }

            _attrs["shared_memory_capacity_bytes"] = sharedResource.CapacityBytes;
            _attrs["shared_memory_managed_arena_limit_bytes"] = sharedSpace.MaxAllocationBytesPerScope;
            _attrs["shared_memory_allocation_size_policy"] = sharedSpace.AllocationSizePolicy switch
            {
                TargetMemoryAllocationSizePolicy.GranularityAligned => "granularity_aligned",
                TargetMemoryAllocationSizePolicy.PowerOfTwo => "power_of_two",
                _ => throw new ArgumentOutOfRangeException(nameof(sharedSpace), sharedSpace.AllocationSizePolicy, "Unknown shared-memory allocation size policy."),
            };
            _attrs["shared_memory_allocation_granularity_bytes"] = sharedResource.AllocationGranularityBytes;
            _attrs["forbid_spills"] = true;
        }

        private static long GetMemoryLocationPoolSizeBytes(BaseExpr expr, MemoryLocation location)
        {
            var allocations = new List<TIR.PhysicalBuffer>();
            CollectMemoryLocationAllocations(expr, location, allocations);
            if (allocations.Count == 0)
            {
                return 0L;
            }

            return allocations
                .Distinct((IEqualityComparer<TIR.PhysicalBuffer>)ReferenceEqualityComparer.Instance)
                .Select(buffer => checked(
                    PyNTTKernelSourceConvertVisitor.GetFixedDimension(buffer.Start, $"{location} physical buffer offset") +
                    PyNTTKernelSourceConvertVisitor.GetFixedDimension(buffer.Size, $"{location} physical buffer size")))
                .Max();
        }

        private static void CollectMemoryLocationAllocations(BaseExpr expr, MemoryLocation location, List<TIR.PhysicalBuffer> allocations)
        {
            if (expr is BaseFunction)
            {
                return;
            }

            if (expr is TIR.Buffer buffer && buffer.MemSpan.Buffer.Location == location)
            {
                allocations.Add(buffer.MemSpan.Buffer);
            }

            foreach (var operand in expr.Operands)
            {
                CollectMemoryLocationAllocations(operand, location, allocations);
            }
        }

        private BufferRef ResolveFormalAbiBufferRef(TIR.Buffer buffer, string baseName)
        {
            if (!TryGetFormalTensorStorageNames(buffer, out var poolStrideName, out var poolScopeSizeName))
            {
                throw new NotSupportedException($"PyNTT formal tensor buffer {buffer.Name} is missing backing-pool ABI metadata.");
            }

            return new(
                baseName,
                GetBufferSpanOffsetElements(buffer),
                poolStrideName,
                BuildScopeIndexExpression("shard_index", poolScopeSizeName),
                null,
                false,
                poolScopeSizeName);
        }

        private bool TryGetFormalTensorBaseName(TIR.Buffer buffer, out string baseName)
        {
            baseName = string.Empty;
            if (buffer.MemSpan.Buffer.Start is IVar parameter &&
                _formalTensorParameterBaseNames.TryGetValue(parameter, out var parameterBaseName))
            {
                baseName = parameterBaseName;
                return true;
            }

            return false;
        }

        private bool TryGetFormalTensorStorageNames(TIR.Buffer buffer, out string poolStrideName, out string poolScopeSizeName)
        {
            poolStrideName = string.Empty;
            poolScopeSizeName = string.Empty;
            if (buffer.MemSpan.Buffer.Start is IVar parameter &&
                _formalTensorParameterPoolStrideNames.TryGetValue(parameter, out var parameterPoolStrideName) &&
                _formalTensorParameterPoolScopeSizeNames.TryGetValue(parameter, out var parameterPoolScopeSizeName))
            {
                poolStrideName = parameterPoolStrideName;
                poolScopeSizeName = parameterPoolScopeSizeName;
                return true;
            }

            return false;
        }

        private bool TryGetFormalTensorDimensions(TIR.Buffer buffer, out PyNTTDimExpression[] dimensions)
        {
            dimensions = Array.Empty<PyNTTDimExpression>();
            if (buffer.MemSpan.Buffer.Start is IVar parameter &&
                UsesBackingTensorLogicalLayout(buffer, parameter) &&
                _formalTensorParameterDimensions.TryGetValue(parameter, out var parameterDimensions))
            {
                dimensions = parameterDimensions;
                return true;
            }

            return false;
        }

        private bool TryGetFormalTensorGlobalOffsets(TIR.Buffer buffer, out PyNTTDimExpression[] offsets)
        {
            offsets = Array.Empty<PyNTTDimExpression>();
            if (buffer.MemSpan.Buffer.Start is IVar parameter &&
                UsesBackingTensorLogicalLayout(buffer, parameter) &&
                _formalTensorParameterGlobalOffsets.TryGetValue(parameter, out var parameterOffsets))
            {
                offsets = parameterOffsets;
                return true;
            }

            return false;
        }

        private bool TryGetFormalTensorSourceSplitAxes(TIR.Buffer buffer, out int[][] splitAxes)
        {
            splitAxes = Array.Empty<int[]>();
            if (buffer.MemSpan.Buffer.Start is IVar parameter &&
                UsesBackingTensorLogicalLayout(buffer, parameter) &&
                _formalTensorParameterSourceSplitAxes.TryGetValue(parameter, out var parameterSplitAxes))
            {
                splitAxes = parameterSplitAxes;
                return true;
            }

            return false;
        }

        private bool IsFormalTensorBuffer(TIR.Buffer buffer)
            => TryGetFormalTensorBaseName(buffer, out _);

        private string GetDataBaseName(TIR.Buffer buffer)
            => _dataBaseNameByBuffer.TryGetValue(buffer, out var baseName) ? baseName : _dataBaseName;

        private string GetChipLocalDataBaseName(TIR.Buffer buffer)
            => _chipLocalDataBaseNameByBuffer.TryGetValue(buffer, out var baseName) ? baseName : _chipLocalDataBaseName;

        private string GetBlockLocalDataBaseName(TIR.Buffer buffer)
            => _blockLocalDataBaseNameByBuffer.TryGetValue(buffer, out var baseName) ? baseName : _blockLocalDataBaseName;

        private static string AddFixedOffsetBytes(string expression, long offsetBytes)
        {
            if (offsetBytes == 0)
            {
                return expression;
            }

            var offset = offsetBytes.ToString(CultureInfo.InvariantCulture);
            return IsZeroOffset(expression) ? offset : $"(({offset}) + ({expression}))";
        }

        private BufferRef ResolveAbiBufferRef(TIR.Buffer buffer, string baseName, bool registerAbiViewStrideArgs = true)
        {
            var spanOffsetElements = GetBufferSpanOffsetElements(buffer);
            if (buffer.DistributedType is not { })
            {
                return new(baseName, spanOffsetElements, "0", null, null, false);
            }

            var poolStrideElements = $"{baseName}{PoolStrideElementsSuffix}";
            if (!registerAbiViewStrideArgs)
            {
                _extraWorkspaceBaseNames.Add(poolStrideElements);
            }

            var localOffsetElements = GetDistributedCompactLocalOffsetElements(buffer);
            var offsetElements = spanOffsetElements;
            if (!IsZeroOffset(localOffsetElements))
            {
                offsetElements = AddOffsetExpressions(offsetElements, $"tl.where({poolStrideElements} == 0, {localOffsetElements}, 0)");
            }

            var shardCoordHierarchy = RequiresShardCoords(offsetElements)
                ? GetShardCoordHierarchy(buffer)
                : null;
            return new(baseName, offsetElements, poolStrideElements, "shard_index", shardCoordHierarchy, false);
        }

        private string BuildPointerExpression(BufferRef bufferRef, string tritonDType)
        {
            if (bufferRef.SharedAllocation is { } sharedAllocation)
            {
                if (bufferRef.PoolStrideBytes != "0" ||
                    bufferRef.IndexExpression is not null ||
                    bufferRef.IsByteAddressed ||
                    bufferRef.OffsetBytes != "0")
                {
                    throw new NotSupportedException(
                        $"PyNTT shared descriptor {sharedAllocation.DescriptorName} cannot use scalar, byte-addressed, indexed, or pooled pointer arithmetic.");
                }

                if (tritonDType != sharedAllocation.TritonDType)
                {
                    throw new NotSupportedException(
                        $"PyNTT shared descriptor {sharedAllocation.DescriptorName} has element type {sharedAllocation.TritonDType}, " +
                        $"but codegen requested {tritonDType}. A dtype reinterpretation must be represented by a typed TIR buffer view.");
                }

                return sharedAllocation.DescriptorName;
            }

            var expression = bufferRef.BaseName;
            if (!string.IsNullOrWhiteSpace(bufferRef.IndexExpression) && bufferRef.PoolStrideBytes != "0")
            {
                expression += $" + {bufferRef.IndexExpression} * {bufferRef.PoolStrideBytes}";
            }

            if (!IsZeroOffset(bufferRef.OffsetBytes))
            {
                expression += $" + {bufferRef.OffsetBytes}";
            }

            var pointerType = GetPointerTypeExpression(tritonDType, bufferRef.AddressSpace);
            return $"({expression}).to({pointerType})";
        }

        private static string GetPointerTypeExpression(string tritonDType, int addressSpace)
            => addressSpace == 1
                ? $"tl.pointer_type({tritonDType})"
                : $"tl.pointer_type({tritonDType}, {addressSpace.ToString(CultureInfo.InvariantCulture)})";

        private string BuildHelperCall(string helperName, params string[] leadingArguments)
        {
            var helperArguments = _helperArguments.TryGetValue(helperName, out var arguments)
                ? arguments
                : Array.Empty<string>();
            var helperScalarArguments = _helperScalarArguments.TryGetValue(helperName, out var scalarArguments)
                ? scalarArguments
                : Array.Empty<string>();
            var callArguments = leadingArguments.Concat(helperArguments).ToArray();
            _helperCalls.Add(new(helperName, callArguments));
            var args = leadingArguments
                .Select(FormatHelperCallArgument)
                .Concat(helperArguments)
                .Concat(GetCurrentWorkspaceParameterNames())
                .Concat(helperScalarArguments)
                .Concat(new[] { "block_size" });
            return $"{helperName}({string.Join(", ", args)})";
        }

        private void WriteHelperInvocation(string helperName, params string[] leadingArguments)
        {
            WriteLine(BuildHelperCall(helperName, leadingArguments));
        }

        private string[] GetCurrentWorkspaceParameterNames()
            =>
            [
                _dataBaseName,
                "rdata",
                "chip_local_rdata",
                _chipLocalDataBaseName,
                "block_local_rdata",
                _blockLocalDataBaseName,
                "data_pool_stride_bytes",
                "block_local_data_pool_stride_bytes",
            ];

        private static string BuildDeviceFunctionCallPlaceholder(string functionName, IReadOnlyList<string>? extraArguments = null)
            => $"{DeviceFunctionCallPrefix}{functionName}({string.Join(", ", extraArguments ?? Array.Empty<string>())})";

        private static string BuildRawPythonArgument(string expression) => $"py:{expression}";

        private static string FormatHelperCallArgument(string argument)
            => argument.StartsWith("py:", StringComparison.Ordinal) ? argument[3..] : argument;

        private PyNTTDimExpression[] GetBufferShape(TIR.Buffer buffer)
        {
            if (_bufferActiveShapeOverrides.TryGetValue(buffer, out var activeShape))
            {
                return activeShape;
            }

            if (buffer.DistributedType is { } distributedType)
            {
                return GetRankedShapeDimensions(DistributedUtility.GetDividedTensorType(distributedType).Shape, $"{buffer.Name} distributed local shape")
                    .Select(GetDimensionExpression)
                    .ToArray();
            }

            return buffer.Dimensions.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
        }

        private PyNTTDimExpression[] GetBufferActiveShape(TIR.Buffer buffer)
        {
            if (_bufferActiveShapeOverrides.TryGetValue(buffer, out var overriddenActiveShape))
            {
                return overriddenActiveShape;
            }

            if (TryGetFormalTensorDimensions(buffer, out var dimensions))
            {
                return dimensions;
            }

            if (buffer.DistributedType is { } distributedType)
            {
                var shardIndex = Enumerable.Range(0, distributedType.Placement.Rank)
                    .Select(axis => (Dimension)new DimVar($"{ShardCoordDimPrefix}{axis}"))
                    .ToArray();
                var (_, activeShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                var hierarchy = distributedType.Placement.Hierarchy.ToArray();
                return activeShape
                    .Select(dimension => GetLocalRegionDimensionExpression(dimension, hierarchy))
                    .ToArray();
            }

            return buffer.Dimensions.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
        }

        private PyNTTDimExpression[] GetBufferGlobalShape(TIR.Buffer buffer)
        {
            if (_bufferGlobalShapeOverrides.TryGetValue(buffer, out var overriddenGlobalShape))
            {
                return overriddenGlobalShape;
            }

            if (buffer.DistributedType is { } distributedType)
            {
                return GetRankedShapeDimensions(distributedType.TensorType.Shape, $"{buffer.Name} distributed global shape")
                    .Select(GetDimensionExpression)
                    .ToArray();
            }

            return buffer.Dimensions.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
        }

        private PyNTTDimExpression[] GetBufferGlobalOffsets(TIR.Buffer buffer)
        {
            if (_bufferGlobalOffsetOverrides.TryGetValue(buffer, out var offsets))
            {
                return offsets;
            }

            if (TryGetFormalTensorGlobalOffsets(buffer, out offsets))
            {
                return offsets;
            }

            if (buffer.DistributedType is { } distributedType)
            {
                var shardIndex = Enumerable.Range(0, distributedType.Placement.Rank)
                    .Select(axis => (Dimension)new DimVar($"{ShardCoordDimPrefix}{axis}"))
                    .ToArray();
                var (localOffset, _) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                var hierarchy = distributedType.Placement.Hierarchy.ToArray();
                return localOffset
                    .Select(dimension => GetLocalRegionDimensionExpression(dimension, hierarchy))
                    .ToArray();
            }

            return Enumerable.Range(0, buffer.Rank)
                .Select(_ => PyNTTDimExpression.Zero)
                .ToArray();
        }

        private PyNTTDimExpression[] GetBufferStrides(TIR.Buffer buffer)
        {
            // BufferSubview and AllocateBufferView select a region from an existing
            // logical buffer; they do not define a new layout. Follow the view edge
            // so formal ABI strides (including non-contiguous sharded const views)
            // remain visible to the kernel. A real layout change is represented by
            // the source TIR.Buffer itself and therefore terminates this recursion.
            if (_bufferViewSourceByBuffer.TryGetValue(buffer, out var viewSource))
            {
                return GetBufferStrides(viewSource.Source);
            }

            if (TryGetFormalTensorBaseName(buffer, out var formalBaseName) &&
                buffer.MemSpan.Buffer.Start is BufferVar formalParameter &&
                UsesBackingTensorLogicalLayout(buffer, formalParameter))
            {
                return formalParameter.LayoutAnnotation.Kind switch
                {
                    BufferLayoutKind.ExactStrided => buffer.Strides.ToArray().Select(GetDimensionExpression).ToArray(),
                    BufferLayoutKind.RuntimeStrided => GetAbiBufferStrideExpressions(formalBaseName, buffer, registerAbiViewStrideArgs: false),
                    _ => throw new NotSupportedException($"PyNTT formal tensor {formalParameter.Name} cannot use layout {formalParameter.LayoutAnnotation}."),
                };
            }

            if (buffer.MemSpan.Buffer.Location == MemoryLocation.Input &&
                buffer.MemSpan.Buffer.Start is BufferVar inputParameter &&
                UsesBackingTensorLogicalLayout(buffer, inputParameter))
            {
                if (inputParameter.LayoutAnnotation.Kind == BufferLayoutKind.ExactStrided)
                {
                    return buffer.Strides.ToArray().Select(GetDimensionExpression).ToArray();
                }

                if (inputParameter.LayoutAnnotation.Kind == BufferLayoutKind.RuntimeStrided)
                {
                    var inputIndex = GetInputIndex(buffer).ToString(CultureInfo.InvariantCulture);
                    return GetAbiBufferStrideExpressions($"input{inputIndex}", buffer);
                }

                throw new NotSupportedException($"PyNTT input tensor {inputParameter.Name} cannot use layout {inputParameter.LayoutAnnotation}.");
            }

            if (buffer.MemSpan.Buffer.Location == MemoryLocation.Output &&
                buffer.MemSpan.Buffer.Start is BufferVar outputParameter &&
                UsesBackingTensorLogicalLayout(buffer, outputParameter))
            {
                if (outputParameter.LayoutAnnotation.Kind == BufferLayoutKind.ExactStrided)
                {
                    return buffer.Strides.ToArray().Select(GetDimensionExpression).ToArray();
                }

                if (outputParameter.LayoutAnnotation.Kind == BufferLayoutKind.RuntimeStrided)
                {
                    var outputIndex = GetOutputIndex(buffer).ToString(CultureInfo.InvariantCulture);
                    return GetAbiBufferStrideExpressions($"output{outputIndex}", buffer);
                }

                throw new NotSupportedException($"PyNTT output tensor {outputParameter.Name} cannot use layout {outputParameter.LayoutAnnotation}.");
            }

            return buffer.Strides.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
        }

        private PyNTTDimExpression[] GetAbiBufferStrideExpressions(string prefix, TIR.Buffer buffer, bool registerAbiViewStrideArgs = true)
        {
            var stridePrefix = GetVectorLaneElementCount(buffer.ElemType) == 1
                ? $"{prefix}_scalar_stride"
                : $"{prefix}_stride";
            var strides = new PyNTTDimExpression[buffer.Rank];
            for (var axis = 0; axis < buffer.Rank; axis++)
            {
                var name = $"{stridePrefix}{axis.ToString(CultureInfo.InvariantCulture)}";
                if (registerAbiViewStrideArgs)
                {
                    _abiViewStrideArgNames.Add(name);
                }
                else
                {
                    _extraWorkspaceBaseNames.Add(name);
                }

                strides[axis] = new PyNTTDimExpression(name, name);
            }

            return strides;
        }

        private string GetBufferOffsetBytes(TIR.Buffer buffer)
            => GetBufferOffsetDimension(buffer).TritonExpression;

        private PyNTTDimExpression GetBufferOffsetDimension(TIR.Buffer buffer)
        {
            var physicalOffset = GetPhysicalBufferStartOffset(buffer.MemSpan.Buffer.Start, $"{buffer.MemSpan.Buffer.Location} physical buffer offset");
            var spanOffset = GetLocalRegionDimensionExpression(buffer.MemSpan.Start, GetShardCoordHierarchy(buffer));
            return AddDimExpression(physicalOffset, spanOffset);
        }

        private PyNTTDimExpression GetPhysicalBufferStartOffset(BaseExpr expr, string name)
        {
            expr = UnwrapInputBoxing(expr);
            if (expr is BufferVar bufferVar && _formalWorkspaceParameters.Contains(bufferVar))
            {
                return PyNTTDimExpression.Zero;
            }

            return GetDimensionExpression(expr, name);
        }

        private string GetBufferSpanOffsetBytes(TIR.Buffer buffer)
        {
            var spanOffset = GetLocalRegionDimensionExpression(buffer.MemSpan.Start, GetShardCoordHierarchy(buffer));
            return spanOffset.TritonExpression;
        }

        private string GetBufferSpanOffsetElements(TIR.Buffer buffer)
        {
            var spanOffset = GetBufferSpanOffsetBytes(buffer);
            return DivideOffsetExpression(spanOffset, GetScalarElementSizeBytes(buffer.ElemType));
        }

        private string GetDistributedCompactLocalOffsetElements(TIR.Buffer buffer)
        {
            if (buffer.DistributedType is not { } distributedType)
            {
                return "0";
            }

            var shardIndex = Enumerable.Range(0, distributedType.Placement.Rank)
                .Select(axis => (Dimension)new DimVar($"{ShardCoordDimPrefix}{axis}"))
                .ToArray();
            var (localOffset, _) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
            var globalShape = GetRankedShapeDimensions(distributedType.TensorType.Shape, $"{buffer.Name} distributed global shape").ToArray();
            var globalStrides = TensorUtilities.GetDefaultStrides(globalShape);
            var offsetElements = TensorUtilities.GetLinearOffset(globalStrides, localOffset) * GetVectorLaneElementCount(buffer.ElemType);
            return GetLocalRegionDimensionExpression(offsetElements, distributedType.Placement.Hierarchy.ToArray()).TritonExpression;
        }

        private int[] GetHierarchy(TIR.Buffer buffer)
        {
            return buffer.DistributedType is { } distributedType
                ? distributedType.Placement.Hierarchy.ToArray()
                : GetBlockHierarchy(_targetOptions);
        }

        private static int[][] GetBufferSplitAxes(TIR.Buffer buffer, int rank)
        {
            if (buffer.DistributedType is not { } distributedType)
            {
                return CreateEmptySplitAxes(rank);
            }

            return GetSplitAxes(distributedType);
        }

        private int[][] GetBufferSourceSplitAxes(TIR.Buffer buffer, int rank)
        {
            if (_bufferSourceSplitAxesOverrides.TryGetValue(buffer, out var splitAxes))
            {
                return ValidateSplitAxesRank(splitAxes, rank, $"{buffer.Name} source split axes");
            }

            if (TryGetFormalTensorSourceSplitAxes(buffer, out splitAxes))
            {
                return ValidateSplitAxesRank(splitAxes, rank, $"{buffer.Name} formal source split axes");
            }

            return GetBufferSplitAxes(buffer, rank);
        }

        private static int[][] CreateEmptySplitAxes(int rank)
            => Enumerable.Range(0, rank).Select(_ => Array.Empty<int>()).ToArray();

        private static PyNTTDimExpression[] CreateZeroDimExpressions(int rank)
            => Enumerable.Range(0, rank).Select(_ => PyNTTDimExpression.Zero).ToArray();

        private static int[][] CloneSplitAxes(IReadOnlyList<int[]> splitAxes)
            => splitAxes.Select(axis => axis.ToArray()).ToArray();

        private static int[][] ValidateSplitAxesRank(int[][] splitAxes, int rank, string context)
        {
            if (splitAxes.Length != rank)
            {
                throw new NotSupportedException($"PyNTT {context} rank mismatch: expected {rank}, got {splitAxes.Length}.");
            }

            return splitAxes;
        }

        private static int[] GetMatmulReduceAxes(TIR.Buffer lhs, TIR.Buffer rhs, TIR.Buffer output, Nncase.IR.Math.MatMulDimInfo dimInfo)
        {
            if (lhs.DistributedType is not { } lhsType ||
                dimInfo.Lk >= lhsType.AxisPolicies.Count ||
                lhsType.AxisPolicies[dimInfo.Lk] is not SBPSplit lhsSplit)
            {
                return Array.Empty<int>();
            }

            var reduceAxes = lhsSplit.Axes.ToArray();
            if (reduceAxes.Length == 0)
            {
                return Array.Empty<int>();
            }

            if (reduceAxes.Any(axis => axis < 0 || axis >= lhsType.Placement.Rank))
            {
                throw new NotSupportedException($"PyNTT Matmul K-axis split uses placement axes [{string.Join(",", reduceAxes)}] outside placement rank {lhsType.Placement.Rank}.");
            }

            if (rhs.DistributedType is not { } rhsType ||
                dimInfo.Rk >= rhsType.AxisPolicies.Count ||
                rhsType.AxisPolicies[dimInfo.Rk] is not SBPSplit rhsSplit ||
                !rhsSplit.Axes.ToArray().SequenceEqual(reduceAxes))
            {
                throw new NotSupportedException("PyNTT Matmul K-axis split expects lhs and rhs K axes to use the same placement split axes.");
            }

            if (output.DistributedType is { } outputType &&
                outputType.AxisPolicies
                    .OfType<SBPSplit>()
                    .Any(split => split.Axes.Any(axis => reduceAxes.Contains(axis))))
            {
                throw new NotSupportedException("PyNTT Matmul K-axis split cannot write an output that is also split by the K reduction placement axes.");
            }

            return reduceAxes;
        }

        private PyNTTDimExpression GetDimensionExpression(Dimension dimension) => _dimEmitter.Emit(dimension);

        private static PyNTTDimExpression AddDimExpression(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
        {
            if (lhs.FixedValue == 0)
            {
                return rhs;
            }

            if (rhs.FixedValue == 0)
            {
                return lhs;
            }

            var fixedValue = lhs.FixedValue.HasValue && rhs.FixedValue.HasValue
                ? checked(lhs.FixedValue.Value + rhs.FixedValue.Value)
                : (long?)null;
            var minValue = lhs.MinValue.HasValue && rhs.MinValue.HasValue
                ? checked(lhs.MinValue.Value + rhs.MinValue.Value)
                : (long?)null;
            var maxValue = lhs.MaxValue.HasValue && rhs.MaxValue.HasValue
                ? checked(lhs.MaxValue.Value + rhs.MaxValue.Value)
                : (long?)null;
            var expression = new PyNTTDimExpression(
                $"(({lhs.PythonExpression}) + ({rhs.PythonExpression}))",
                $"(({lhs.TritonExpression}) + ({rhs.TritonExpression}))",
                fixedValue,
                minValue,
                maxValue);
            return WithAffineEquivalence(expression, lhs, rhs, subtract: false);
        }

        private static PyNTTDimExpression SubtractDimExpression(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
        {
            if (rhs.FixedValue == 0)
            {
                return lhs;
            }

            var fixedValue = lhs.FixedValue.HasValue && rhs.FixedValue.HasValue
                ? checked(lhs.FixedValue.Value - rhs.FixedValue.Value)
                : (long?)null;
            var minValue = lhs.MinValue.HasValue && rhs.MaxValue.HasValue
                ? checked(lhs.MinValue.Value - rhs.MaxValue.Value)
                : (long?)null;
            var maxValue = lhs.MaxValue.HasValue && rhs.MinValue.HasValue
                ? checked(lhs.MaxValue.Value - rhs.MinValue.Value)
                : (long?)null;
            var expression = new PyNTTDimExpression(
                $"(({lhs.PythonExpression}) - ({rhs.PythonExpression}))",
                $"(({lhs.TritonExpression}) - ({rhs.TritonExpression}))",
                fixedValue,
                minValue,
                maxValue);
            return WithAffineEquivalence(expression, lhs, rhs, subtract: true);
        }

        private static PyNTTDimExpression MultiplyDimExpressions(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
        {
            if (lhs.FixedValue == 0 || rhs.FixedValue == 0)
            {
                return PyNTTDimExpression.Zero;
            }

            if (lhs.FixedValue == 1)
            {
                return rhs;
            }

            if (rhs.FixedValue == 1)
            {
                return lhs;
            }

            if (lhs.FixedValue is { } lhsFixed)
            {
                return MultiplyDim(rhs, lhsFixed);
            }

            if (rhs.FixedValue is { } rhsFixed)
            {
                return MultiplyDim(lhs, rhsFixed);
            }

            return new PyNTTDimExpression(
                $"(({lhs.PythonExpression}) * ({rhs.PythonExpression}))",
                $"(({lhs.TritonExpression}) * ({rhs.TritonExpression}))")
                .EnsureEquivalence();
        }

        private PyNTTDimExpression GetLocalRegionDimensionExpression(Dimension dimension, IReadOnlyList<int> hierarchy)
        {
            var emitter = new PyNTTDimExpressionEmitter(
                RegisterLocalRegionRuntimeScalar,
                name => FormatLocalRegionRuntimeScalar(name, hierarchy),
                BuildThreadIdExpression(_targetOptions),
                ResolveActiveLoopVariable);
            return emitter.Emit(dimension);
        }

        private PyNTTDimExpression GetDimensionExpression(BaseExpr expr, string name)
        {
            expr = ResolveBoundExpression(expr);
            return expr switch
            {
                None => PyNTTDimExpression.Zero,
                Dimension dimension => _dimEmitter.Emit(dimension),
                TensorConst tensorConst when tensorConst.Value.Shape.IsScalar => FormatDimensionConst(tensorConst),
                _ => throw new NotSupportedException($"PyNTT requires scalar dimension expression for {name}, got {expr.GetType().Name}."),
            };
        }

        private static PyNTTDimExpression FormatDimensionConst(TensorConst tensorConst)
        {
            var value = ReadScalarInt64(tensorConst.Value, "PyNTT dimension constant");
            return ToDim(value);
        }

        private static string AddOffsetExpressions(string lhs, string rhs)
        {
            if (IsZeroOffset(lhs))
            {
                return rhs;
            }

            if (IsZeroOffset(rhs))
            {
                return lhs;
            }

            return $"(({lhs}) + ({rhs}))";
        }

        private static string DivideOffsetExpression(string expression, int divisor)
        {
            if (divisor <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(divisor), divisor, "Offset divisor must be positive.");
            }

            if (divisor == 1 || IsZeroOffset(expression))
            {
                return expression;
            }

            if (long.TryParse(expression.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value))
            {
                if (value % divisor != 0)
                {
                    throw new NotSupportedException($"PyNTT ABI buffer byte offset {value} is not aligned to scalar element size {divisor}.");
                }

                return (value / divisor).ToString(CultureInfo.InvariantCulture);
            }

            return $"(({expression}) // {divisor.ToString(CultureInfo.InvariantCulture)})";
        }

        private static string MultiplyOffsetExpression(string expression, int multiplier)
        {
            if (multiplier <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(multiplier), multiplier, "Offset multiplier must be positive.");
            }

            if (multiplier == 1 || IsZeroOffset(expression))
            {
                return expression;
            }

            if (long.TryParse(expression.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value))
            {
                return checked(value * multiplier).ToString(CultureInfo.InvariantCulture);
            }

            return $"(({expression}) * {multiplier.ToString(CultureInfo.InvariantCulture)})";
        }

        private static bool IsZeroOffset(string expression)
            => string.Equals(expression.Trim(), "0", StringComparison.Ordinal);

        private static bool RequiresShardCoords(string expression)
            => expression.Contains("shard_coord", StringComparison.Ordinal);

        private int[] GetShardCoordHierarchy(TIR.Buffer buffer)
            => buffer.DistributedType is { } distributedType
                ? distributedType.Placement.Hierarchy.ToArray()
                : GetBlockHierarchy(_targetOptions);

        private static string BuildThreadIdExpression(PyNTTTargetOptions targetOptions)
            => "0";

        private static string BuildBlockLocalDataIndexExpression(PyNTTTargetOptions targetOptions)
            => BuildScopeIndexExpression(GetBlockLocalDataScopeSize(targetOptions));

        private static int GetBlockLocalDataScopeCount(PyNTTTargetOptions targetOptions)
            => GetScopeCount(targetOptions, GetBlockLocalDataScopeSize(targetOptions));

        private static string BuildScopeIndexExpression(int scopeSize)
            => BuildScopeIndexExpression("shard_index", scopeSize);

        private static string BuildScopeIndexExpression(string linearIndexExpression, int scopeSize)
            => scopeSize <= 1
                ? linearIndexExpression
                : $"(({linearIndexExpression}) // {scopeSize.ToString(CultureInfo.InvariantCulture)})";

        private static string BuildScopeIndexExpression(string linearIndexExpression, string scopeSizeExpression)
            => string.Equals(scopeSizeExpression, "1", StringComparison.Ordinal)
                ? linearIndexExpression
                : $"(({linearIndexExpression}) // ({scopeSizeExpression}))";

        private static int GetScopeCount(PyNTTTargetOptions targetOptions, int scopeSize)
        {
            var hierarchy = GetBlockHierarchy(targetOptions);
            var shardCount = hierarchy.Aggregate(1, (product, value) => checked(product * value));
            return Math.Max(1, shardCount / Math.Max(1, scopeSize));
        }

        private static int GetBlockLocalDataScopeSize(PyNTTTargetOptions targetOptions)
            => 1;

        private static int GetScalarElementSizeBytes(DataType dataType)
        {
            return dataType is VectorType vectorType
                ? vectorType.ElemType.SizeInBytes
                : dataType.SizeInBytes;
        }

        private void RegisterRuntimeScalar(string name)
        {
            if (_formalDimParameterNames.ContainsKey(name))
            {
                return;
            }

            _helperScalarNameCandidates.Add(name);
            if (!IsActiveLocalScalar(name))
            {
                _runtimeScalarNames.Add(name);
            }
        }

        private string FormatRuntimeScalar(string name)
            => _formalDimParameterNames.TryGetValue(name, out var formalName) ? formalName : name;

        private void PushLocalScalar(string name)
        {
            _helperScalarNameCandidates.Add(name);
            _activeLocalScalarNames.TryGetValue(name, out var count);
            _activeLocalScalarNames[name] = count + 1;
        }

        private void PopLocalScalar(string name)
        {
            if (!_activeLocalScalarNames.TryGetValue(name, out var count))
            {
                throw new InvalidOperationException($"PyNTT local scalar scope for {name} is unbalanced.");
            }

            if (count == 1)
            {
                _activeLocalScalarNames.Remove(name);
            }
            else
            {
                _activeLocalScalarNames[name] = count - 1;
            }
        }

        private bool IsActiveLocalScalar(string name)
            => _activeLocalScalarNames.ContainsKey(name);

        private void PushLocalBuffer(string name)
        {
            _activeLocalBufferNames.TryGetValue(name, out var count);
            _activeLocalBufferNames[name] = count + 1;
        }

        private void PopLocalBuffer(string name)
        {
            if (!_activeLocalBufferNames.TryGetValue(name, out var count))
            {
                throw new InvalidOperationException($"PyNTT local buffer scope for {name} is unbalanced.");
            }

            if (count == 1)
            {
                _activeLocalBufferNames.Remove(name);
            }
            else
            {
                _activeLocalBufferNames[name] = count - 1;
            }
        }

        private void PushLoopVariableRange(string name, PyNTTDimExpression range)
        {
            if (!_activeLoopVariableRanges.TryGetValue(name, out var ranges))
            {
                ranges = new Stack<PyNTTDimExpression>();
                _activeLoopVariableRanges.Add(name, ranges);
            }

            ranges.Push(range);
        }

        private void PopLoopVariableRange(string name)
        {
            if (!_activeLoopVariableRanges.TryGetValue(name, out var ranges) || ranges.Count == 0)
            {
                throw new InvalidOperationException($"PyNTT loop variable range scope for {name} is unbalanced.");
            }

            ranges.Pop();
            if (ranges.Count == 0)
            {
                _activeLoopVariableRanges.Remove(name);
            }
        }

        private PyNTTDimExpression? ResolveActiveLoopVariable(DimVar variable)
        {
            var name = SanitizePythonIdentifier(variable.Name);
            return _activeLoopVariableRanges.TryGetValue(name, out var ranges) && ranges.Count > 0
                ? ranges.Peek()
                : null;
        }

        private static PyNTTDimExpression GetLoopVariableRange(
            string name,
            PyNTTDimExpression start,
            PyNTTDimExpression stop,
            PyNTTDimExpression step)
        {
            if (step.FixedValue is not > 0 ||
                !start.MinValue.HasValue ||
                !start.MaxValue.HasValue ||
                !stop.MaxValue.HasValue)
            {
                return new(name, name);
            }

            var min = start.MinValue.Value;
            long max;
            if (start.FixedValue is { } fixedStart)
            {
                var stopMax = stop.MaxValue.Value;
                max = stopMax <= fixedStart
                    ? fixedStart
                    : checked(fixedStart + ((stopMax - 1 - fixedStart) / step.FixedValue.Value * step.FixedValue.Value));
            }
            else
            {
                max = Math.Max(start.MaxValue.Value, checked(stop.MaxValue.Value - 1));
            }

            return min == max
                ? new(name, name, min)
                : new(name, name, null, min, max);
        }

        private void RegisterLocalRegionRuntimeScalar(string name)
        {
            if (!TryGetShardCoordDimAxis(name, out _))
            {
                RegisterRuntimeScalar(name);
            }
        }

        private string FormatLocalRegionRuntimeScalar(string name, IReadOnlyList<int> hierarchy)
            => TryGetShardCoordDimAxis(name, out var axis) ? BuildShardCoordExpression(axis, hierarchy) : FormatRuntimeScalar(name);

        private static string BuildShardCoordExpression(int axis, IReadOnlyList<int> hierarchy)
        {
            if (axis < 0 || axis >= hierarchy.Count)
            {
                throw new NotSupportedException($"PyNTT local shard coordinate axis {axis} is outside hierarchy rank {hierarchy.Count}.");
            }

            var divisor = 1;
            for (var i = axis + 1; i < hierarchy.Count; i++)
            {
                divisor = checked(divisor * hierarchy[i]);
            }

            var dividend = divisor == 1
                ? "shard_index"
                : $"(shard_index // {divisor.ToString(CultureInfo.InvariantCulture)})";
            var extent = hierarchy[axis];
            return extent == 1
                ? "0"
                : $"(({dividend}) % {extent.ToString(CultureInfo.InvariantCulture)})";
        }

        private static bool TryGetShardCoordDimAxis(string name, out int axis)
        {
            axis = default;
            if (!name.StartsWith(ShardCoordDimPrefix, StringComparison.Ordinal))
            {
                return false;
            }

            return int.TryParse(name[ShardCoordDimPrefix.Length..], NumberStyles.None, CultureInfo.InvariantCulture, out axis);
        }

        private string BuildScalarExpression(BaseExpr expr)
        {
            expr = ResolveBoundExpression(expr);
            if (expr is Dimension dimension)
            {
                return _dimEmitter.Emit(dimension).TritonExpression;
            }

            if (expr is TensorConst tensorConst)
            {
                return FormatScalarConst(tensorConst);
            }

            if (expr is IVar parameter && _parameterNames.TryGetValue(parameter, out var parameterName))
            {
                return SanitizePythonIdentifier(parameterName);
            }

            if (expr is Call call)
            {
                var args = call.Arguments.ToArray();
                return call.Target switch
                {
                    AsTensor when args.Length == 1 =>
                        BuildScalarExpression(args[0]),
                    LocalShardDim when args.Length == 1 =>
                        _dimEmitter.Emit(call.AsDim()).TritonExpression,
                    Nncase.IR.Math.Compare compare when args.Length >= 2 =>
                        $"({BuildScalarExpression(args[0])} {GetCompareOperator(compare.CompareOp)} {BuildScalarExpression(args[1])})",
                    Nncase.IR.Math.Binary binary when args.Length >= 2 =>
                        BuildScalarBinaryExpression(binary.BinaryOp, BuildScalarExpression(args[0]), BuildScalarExpression(args[1])),
                    _ => throw new NotSupportedException($"Unsupported PyNTT scalar expression call target: {call.Target.GetType().Name}."),
                };
            }

            throw new NotSupportedException($"Unsupported PyNTT scalar expression: {expr.GetType().Name}.");
        }

        private string GetScalarBoolExpression(BaseExpr expr, string name)
        {
            expr = ResolveBoundExpression(expr);
            if (expr is None)
            {
                return "False";
            }

            if (TryEvaluateScalarBool(expr, out var value))
            {
                return value ? "True" : "False";
            }

            try
            {
                return BuildScalarExpression(expr);
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidCastException)
            {
                throw new NotSupportedException($"PyNTT requires a bool scalar expression for {name}.", ex);
            }
        }

        private static bool TryEvaluateScalarBool(BaseExpr expr, out bool value)
        {
            try
            {
                value = expr.Evaluate().AsTensor().ToScalar<bool>();
                return true;
            }
            catch (Exception ex) when (ex is InvalidCastException or InvalidOperationException or NotSupportedException or ArgumentException)
            {
                value = default;
                return false;
            }
        }

        private Dictionary<Call, ReductionState> CreateReductionScope(For reductionLoop)
            => CreateReductionScope(reductionLoop.Body);

        private Dictionary<Call, ReductionState> CreateReductionScope(params BaseExpr[] bodies)
        {
            var groups = ReductionCodegenUtility.CollectReductionCallGroups(bodies);
            if (groups.Length == 0)
            {
                throw new InvalidOperationException(
                    "PyNTT reduction scope contains no backend reduction operation.");
            }

            var scope = new Dictionary<Call, ReductionState>(ReferenceEqualityComparer.Instance);
            foreach (var group in groups)
            {
                var call = group.Prototype;
                var kind = call.Target switch
                {
                    Nncase.TIR.NTT.Matmul => ReductionKernelKind.Matmul,
                    Nncase.TIR.NTT.PackedMatMul => ReductionKernelKind.PackedMatmul,
                    Nncase.TIR.NTT.QKVParallelLinear => ReductionKernelKind.QKVParallelLinear,
                    Nncase.TIR.NTT.PackedQKVParallelLinear => ReductionKernelKind.PackedQKVParallelLinear,
                    Nncase.TIR.NTT.MatMulGlu => ReductionKernelKind.MatMulGlu,
                    Nncase.TIR.NTT.PackedMatMulGlu => ReductionKernelKind.PackedMatMulGlu,
                    Nncase.TIR.NTT.Reduce => ReductionKernelKind.Reduce,
                    _ => throw new InvalidOperationException(
                        $"Unsupported PyNTT reduction operation {call.Target.GetType().Name}."),
                };
                var logicalAccumulatorCount = ReductionCodegenUtility.GetAccumulatorOperands(call).Length;
                var expectedLogicalAccumulatorCount = kind switch
                {
                    ReductionKernelKind.QKVParallelLinear or ReductionKernelKind.PackedQKVParallelLinear => 3,
                    _ => 1,
                };
                if (logicalAccumulatorCount != expectedLogicalAccumulatorCount)
                {
                    throw new InvalidOperationException(
                        $"PyNTT reduction operation {call.Target.GetType().Name} requires " +
                        $"{expectedLogicalAccumulatorCount} logical reduction-accumulator operands, " +
                        $"but its memory-effect contract declares {logicalAccumulatorCount}.");
                }

                var stateCount = kind switch
                {
                    ReductionKernelKind.MatMulGlu or ReductionKernelKind.PackedMatMulGlu => 2,
                    ReductionKernelKind.Reduce when call.Target is Nncase.TIR.NTT.Reduce { ReduceOp: ReduceOp.Mean } => 2,
                    _ => logicalAccumulatorCount,
                };

                var stateId = _reductionStateCounter++;
                var names = Enumerable.Range(0, stateCount)
                    .Select(index => $"pyntt_reduction_{stateId}_acc{index}")
                    .ToArray();
                var state = new ReductionState(call, kind, names, group.Calls.Length);
                foreach (var groupedCall in group.Calls)
                {
                    scope.Add(groupedCall, state);
                }
            }

            return scope;
        }

        private static IEnumerable<ReductionState> GetDistinctReductionStates(
            IReadOnlyDictionary<Call, ReductionState> scope)
            => new HashSet<ReductionState>(scope.Values, ReferenceEqualityComparer.Instance);

        private bool TryGetReductionState(Call call, out ReductionState state)
        {
            foreach (var scope in _reductionScopes)
            {
                if (scope.TryGetValue(call, out state!))
                {
                    return true;
                }
            }

            state = null!;
            return false;
        }

        private void ConfigureReductionState(
            ReductionState state,
            ReductionAccumulatorInitializer[] initializers,
            string templatePath,
            string finalizeHelperKind,
            Func<string, object> finalizeModelFactory)
        {
            if (state.UpdateCount >= state.ExpectedUpdateCount)
            {
                throw new InvalidOperationException(
                    $"PyNTT reduction operation {state.Call.Target.GetType().Name} exceeds its " +
                    $"{state.ExpectedUpdateCount} planned accumulator updates.");
            }

            const string signatureFunctionName = "pyntt_reduction_finalize_signature";
            var finalizeSignature = JsonSerializer.Serialize(finalizeModelFactory(signatureFunctionName));
            if (state.UpdateCount == 0)
            {
                state.Initializers = initializers;
                state.TemplatePath = templatePath;
                state.FinalizeSignature = finalizeSignature;
                state.FinalizeModel = finalizeModelFactory(GetNextHelperName(finalizeHelperKind));
                return;
            }

            if (state.Initializers is null ||
                !state.Initializers.SequenceEqual(initializers) ||
                !string.Equals(state.TemplatePath, templatePath, StringComparison.Ordinal) ||
                !string.Equals(state.FinalizeSignature, finalizeSignature, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"PyNTT reduction operation {state.Call.Target.GetType().Name} has incompatible " +
                    "accumulator or finalization contracts across structured loop partitions.");
            }
        }

        private static string BuildReductionInitializers(
            IReadOnlyDictionary<Call, ReductionState> scope,
            int indent)
        {
            var builder = new StringBuilder();
            var prefix = new string(' ', indent * 4);
            foreach (var state in GetDistinctReductionStates(scope))
            {
                if (state.UpdateCount != state.ExpectedUpdateCount)
                {
                    throw new InvalidOperationException(
                        $"PyNTT reduction operation {state.Call.Target.GetType().Name} emitted " +
                        $"{state.UpdateCount} of {state.ExpectedUpdateCount} expected accumulator updates.");
                }

                if (state.Initializers is null || state.Initializers.Length != state.Names.Length)
                {
                    throw new InvalidOperationException(
                        $"PyNTT reduction operation {state.Call.Target.GetType().Name} did not configure all accumulator initializers.");
                }

                for (var index = 0; index < state.Names.Length; index++)
                {
                    var name = state.Names[index];
                    var initializer = state.Initializers[index];
                    builder.Append(prefix);
                    builder.Append(name);
                    builder.Append(" = ");
                    builder.Append(initializer.ValueExpression);
                    builder.AppendLine();
                }
            }

            return builder.ToString();
        }

        private void EmitReductionFinalize(ReductionState state)
        {
            if (state.FinalizeModel is null || state.TemplatePath is null)
            {
                throw new InvalidOperationException(
                    $"PyNTT reduction operation {state.Call.Target.GetType().Name} did not configure its finalizer.");
            }

            var helperName = GetHelperFunctionName(state.FinalizeModel);
            WriteHelperTemplate(state.TemplatePath, state.FinalizeModel, requiresInline: true);
            WriteControlLine(BuildHelperCall(
                helperName,
                state.Names.Select(BuildRawPythonArgument).ToArray()));
        }

        private BlockMicroKernelSelection RequireCurrentBlockMicroKernel(string context)
            => _currentBlockMicroKernel
                ?? throw new InvalidOperationException(
                    $"{context} requires the block microkernel contract selected by AutoTiling.");

        private T ApplyBlockMicroKernel<T>(T model, BlockMicroKernelSelection selection)
            where T : IPyNTTBlockMicroKernelTemplateModel
        {
            model.MicroKernelFamily = selection.Family;
            model.MicroKernelVariant = selection.Variant;
            model.MicroKernelParameters = selection.Parameters.ToDictionary(
                pair => pair.Key,
                pair => pair.Value,
                StringComparer.Ordinal);
            if (model is IPyNTTMatrixMicroKernelTemplateModel)
            {
                var contractVersion = GetBlockMicroKernelParameter(selection, TritonBlockMicroKernelContract.VersionParameter);
                if (contractVersion != TritonBlockMicroKernelContract.Version)
                {
                    throw new InvalidOperationException(
                        $"Block microkernel {selection.Family}/{selection.Variant} uses unsupported contract version {contractVersion}; " +
                        $"expected {TritonBlockMicroKernelContract.Version}.");
                }

                _ = GetBlockMicroKernelParameter(selection, TritonBlockMicroKernelContract.InnerMParameter);
                _ = GetBlockMicroKernelParameter(selection, TritonBlockMicroKernelContract.InnerNParameter);
                _ = GetBlockMicroKernelParameter(selection, TritonBlockMicroKernelContract.InnerKParameter);
            }

            var contractKey = string.Join(
                "|",
                model.FunctionName,
                selection.Family,
                selection.Variant,
                string.Join(",", selection.Parameters.OrderBy(pair => pair.Key).Select(pair => $"{pair.Key}={pair.Value}")));
            if (_blockMicroKernelContractKeys.Add(contractKey))
            {
                _blockMicroKernelContracts.Add(new Dictionary<string, object>(StringComparer.Ordinal)
                {
                    ["helper"] = model.FunctionName,
                    ["family"] = selection.Family,
                    ["variant"] = selection.Variant,
                    ["estimated_cycles"] = selection.EstimatedCycles,
                    ["resources"] = selection.Resources,
                    ["parameters"] = selection.Parameters,
                });
            }

            return model;
        }

        private T ApplyCurrentBlockMicroKernelIfPresent<T>(T model)
            where T : IPyNTTBlockMicroKernelTemplateModel
            => _currentBlockMicroKernel is { } selection
                ? ApplyBlockMicroKernel(model, selection)
                : model;

        private static int GetBlockMicroKernelParameter(
            BlockMicroKernelSelection selection,
            string name)
        {
            if (!selection.Parameters.TryGetValue(name, out var value))
            {
                throw new InvalidOperationException(
                    $"Block microkernel {selection.Family}/{selection.Variant} does not define required parameter '{name}'.");
            }

            if (value <= 0 || value > int.MaxValue)
            {
                throw new InvalidOperationException(
                    $"Block microkernel {selection.Family}/{selection.Variant} parameter '{name}' must be in [1, {int.MaxValue}], got {value}.");
            }

            return checked((int)value);
        }

        private static ReductionAccumulatorInitializer BuildMatrixReductionAccumulatorInitializer(
            BlockMicroKernelSelection microKernel,
            bool useGemv,
            int blockM,
            int blockN,
            int nGroupWidth = 1)
        {
            if (microKernel.Variant is not ("register_simt_accumulator" or "register_mma_accumulator"))
            {
                throw new NotSupportedException(
                    $"Unsupported PyNTT matrix microkernel variant {microKernel.Family}/{microKernel.Variant}.");
            }

            if (nGroupWidth <= 0 || blockN % nGroupWidth != 0)
            {
                throw new InvalidOperationException(
                    $"PyNTT matrix accumulator N extent {blockN} must be divisible by grouped width {nGroupWidth}.");
            }

            var accumulatorShape = useGemv &&
                microKernel.Variant == "register_simt_accumulator" &&
                nGroupWidth > 1
                ? $"({blockN / nGroupWidth}, {nGroupWidth})"
                : useGemv
                    ? $"({blockN},)"
                    : $"({blockM}, {blockN})";
            return ReductionAccumulatorInitializer.Register(
                $"tl.zeros({accumulatorShape}, tl.float32)");
        }

        private static int GetReductionBlockExtent(PyNTTDimExpression dimension, string context, int minimum = 1)
        {
            var maximum = dimension.MaxValue
                ?? throw new NotSupportedException($"{context} requires a finite maximum tile extent.");
            if (maximum <= 0)
            {
                throw new NotSupportedException($"{context} requires a positive tile extent, got {maximum}.");
            }

            var extent = Math.Max(maximum, minimum);
            var rounded = System.Numerics.BitOperations.RoundUpToPowerOf2((ulong)extent);
            if (rounded == 0 || rounded > int.MaxValue)
            {
                throw new NotSupportedException($"{context} tile extent {extent} cannot be represented as a Triton tensor dimension.");
            }

            return checked((int)rounded);
        }

        private static int GetMatmulReductionBlockKExtent(
            PyNTTDimExpression dimension,
            string context,
            bool useGemv)
            => GetReductionBlockExtent(dimension, context, useGemv ? 1 : 16);

        private void SetComputeOp(string opKind)
        {
            _opKinds.Add(opKind);
        }

        private void WriteControlLine(string line)
        {
            _body.Append(new string(' ', _bodyIndent * 4));
            _body.AppendLine(line);
        }

        private void WriteTraceMarker(string label)
        {
            _body.Append(new string(' ', _bodyIndent * 4));
            _body.AppendLine($"# pyntt_trace_event: {label}");
        }

        private string GetPrimFunctionCallTraceLabel(string functionName)
        {
            if (!_primFunctionCallCounters.TryGetValue(functionName, out var index))
            {
                index = 0;
            }

            _primFunctionCallCounters[functionName] = index + 1;
            return $"{functionName}#{index.ToString(CultureInfo.InvariantCulture)}";
        }

        private void WriteLine(string line, HelperBarrierKind barrierKind = HelperBarrierKind.None)
        {
            _body.Append(new string(' ', _bodyIndent * 4));
            _body.AppendLine(line);
            WriteBarrier(barrierKind);
        }

        private void WriteExplicitBarrier(TIR.NTT.BarrierScope scope)
        {
            if (scope == TIR.NTT.BarrierScope.Chip)
            {
                _attrs["requires_grid_barrier"] = true;
                WriteBarrier(HelperBarrierKind.Grid);
                return;
            }

            WriteBarrier(HelperBarrierKind.Block);
        }

        private void WriteBarrier(HelperBarrierKind barrierKind)
        {
            if (barrierKind == HelperBarrierKind.None)
            {
                return;
            }

            _body.Append(new string(' ', _bodyIndent * 4));
            _body.AppendLine(barrierKind switch
            {
                HelperBarrierKind.Block => "tl.debug_barrier()",
                HelperBarrierKind.Grid => "tle.distributed_barrier(pyntt_grid_mesh)",
                _ => throw new ArgumentOutOfRangeException(nameof(barrierKind), barrierKind, null),
            });
        }

        private string[] WriteHelperTemplate(string templatePath, object model, bool requiresInline = false)
        {
            if (model is IPyNTTBlockMicroKernelTemplateModel microKernelModel &&
                _currentBlockMicroKernel is { } selection)
            {
                if ((!string.IsNullOrEmpty(microKernelModel.MicroKernelFamily) &&
                     !string.Equals(microKernelModel.MicroKernelFamily, selection.Family, StringComparison.Ordinal)) ||
                    (!string.IsNullOrEmpty(microKernelModel.MicroKernelVariant) &&
                     !string.Equals(microKernelModel.MicroKernelVariant, selection.Variant, StringComparison.Ordinal)))
                {
                    throw new InvalidOperationException(
                        $"PyNTT helper {microKernelModel.FunctionName} already carries block microkernel " +
                        $"{microKernelModel.MicroKernelFamily}/{microKernelModel.MicroKernelVariant}, but the current TIR call selects " +
                        $"{selection.Family}/{selection.Variant}.");
                }

                ApplyBlockMicroKernel(microKernelModel, selection);
            }

            var runtimeShapeArgs = CollectHelperScalarArguments(model, _helperScalarNameCandidates);
            var runtimeShapeArgsProperty = model.GetType().GetProperty("RuntimeShapeArgs");
            runtimeShapeArgsProperty?.SetValue(model, runtimeShapeArgs);

            var arguments = CollectHelperArguments(model)
                .Concat(CollectHelperScalarArguments(model, _extraWorkspaceBaseNames))
                .Concat(CollectHelperScalarArguments(model, _activeLocalBufferNames.Keys))
                .Distinct(StringComparer.Ordinal)
                .ToArray();
            var functionName = GetHelperFunctionName(model);
            _helperArguments[functionName] = arguments;
            _helperScalarArguments[functionName] = runtimeShapeArgs;
            _helpers.Add(new(GetJinjaTemplateName(templatePath), model, arguments, requiresInline));
            return arguments;
        }

        private static string GetHelperFunctionName(object model)
        {
            var property = model.GetType().GetProperty("FunctionName")
                ?? throw new NotSupportedException($"PyNTT helper model {model.GetType().Name} must expose FunctionName.");
            return property.GetValue(model) as string
                ?? throw new NotSupportedException($"PyNTT helper model {model.GetType().Name} has non-string FunctionName.");
        }

        private static string[] CollectHelperArguments(object model)
        {
            var arguments = new HashSet<string>(StringComparer.Ordinal);
            CollectHelperArguments(JsonSerializer.SerializeToNode(model), arguments);
            return arguments
                .OrderBy(argument => argument.StartsWith("input", StringComparison.Ordinal) ? 0 : 1)
                .ThenBy(ParseAbiArgumentIndex)
                .ThenBy(GetAbiArgumentKind)
                .ThenBy(argument => argument, StringComparer.Ordinal)
                .ToArray();
        }

        private static string[] CollectHelperScalarArguments(object model, IEnumerable<string> candidates)
        {
            var text = JsonSerializer.Serialize(model);
            return candidates
                .Where(candidate => ContainsIdentifier(text, candidate))
                .Distinct(StringComparer.Ordinal)
                .OrderBy(candidate => candidate, StringComparer.Ordinal)
                .ToArray();
        }

        private static bool ContainsIdentifier(string text, string identifier)
            => Regex.IsMatch(
                text,
                $@"(?<![A-Za-z0-9_]){Regex.Escape(identifier)}(?![A-Za-z0-9_])",
                RegexOptions.CultureInvariant);

        private static void CollectHelperArguments(JsonNode? node, HashSet<string> arguments)
        {
            switch (node)
            {
                case null:
                    return;
                case JsonValue value when value.TryGetValue<string>(out var text):
                    foreach (Match match in AbiArgumentReferenceRegex.Matches(text))
                    {
                        arguments.Add(match.Value);
                    }

                    return;
                case JsonValue:
                    return;
                case JsonArray array:
                    foreach (var item in array)
                    {
                        CollectHelperArguments(item, arguments);
                    }

                    return;
                case JsonObject obj:
                    foreach (var property in obj)
                    {
                        CollectHelperArguments(property.Value, arguments);
                    }

                    return;
            }
        }

        private static int ParseAbiArgumentIndex(string argument)
        {
            argument = StripAbiArgumentSuffix(argument);
            var indexStart = argument.StartsWith("input", StringComparison.Ordinal) ? "input".Length : "output".Length;
            return int.Parse(argument[indexStart..], CultureInfo.InvariantCulture);
        }

        private static int GetAbiArgumentKind(string argument)
        {
            if (argument.Contains("_stride", StringComparison.Ordinal))
            {
                return 1;
            }

            if (argument.EndsWith(PoolStrideElementsSuffix, StringComparison.Ordinal))
            {
                return 2;
            }

            return 0;
        }

        private static string StripAbiArgumentSuffix(string argument)
            => AbiArgumentSuffixRegex.Replace(argument, string.Empty);

        private static string GetJinjaTemplateName(string templatePath)
        {
            const string prefix = "triton/kernels/";
            const string suffix = ".py.jinja";
            if (!templatePath.StartsWith(prefix, StringComparison.Ordinal) ||
                !templatePath.EndsWith(suffix, StringComparison.Ordinal))
            {
                throw new NotSupportedException($"Unsupported PyNTT Jinja template path: {templatePath}.");
            }

            return templatePath;
        }

        private string GetHelperName(string kind, int index)
        {
            var scopeName = ReferenceEquals(_currentFunction, _function) &&
                _semanticHelperScopes.TryPeek(out var currentScopeName)
                ? currentScopeName
                : null;
            var semanticName = scopeName is null
                ? $"{_ownerName}__{kind}__{index.ToString(CultureInfo.InvariantCulture)}"
                : $"{_ownerName}__{scopeName}__{kind}__{index.ToString(CultureInfo.InvariantCulture)}";
            return SanitizeBoundedPythonIdentifier(semanticName);
        }

        private string GetNextHelperName(string kind)
        {
            var index = _helperCounters.TryGetValue(kind, out var current) ? current : 0;
            _helperCounters[kind] = index + 1;
            return GetHelperName(kind, index);
        }

        private static BaseExpr UnwrapInputBoxing(BaseExpr expr)
        {
            while (expr is Call call && call.Target is Boxing)
            {
                expr = call.Arguments[0];
            }

            return expr;
        }

        public sealed class KernelAbiState
        {
            public KernelAbiState(int outputCount)
                : this(
                    new List<string>(),
                    new List<PyNTTKVCacheFieldInputMetadata>(),
                    new SortedSet<string>(StringComparer.Ordinal),
                    new SortedSet<string>(StringComparer.Ordinal),
                    new Dictionary<TIR.Buffer, int>(ReferenceEqualityComparer.Instance),
                    new Dictionary<BufferVar, TIR.Buffer>(ReferenceEqualityComparer.Instance),
                    new HashSet<int>(),
                    new DistributedType?[outputCount],
                    new Dictionary<int, int>(),
                    new HashSet<PrimFunction>(ReferenceEqualityComparer.Instance),
                    new HashSet<string>(StringComparer.Ordinal),
                    new Dictionary<PrimFunction, DeviceFunctionDefinition>(ReferenceEqualityComparer.Instance),
                    new Dictionary<string, DeviceFunctionDefinition>(StringComparer.Ordinal))
            {
            }

            public KernelAbiState(
                List<string> inputNames,
                List<PyNTTKVCacheFieldInputMetadata> kvCacheFieldInputs,
                SortedSet<string> runtimeScalarNames,
                SortedSet<string> abiViewStrideArgNames,
                Dictionary<TIR.Buffer, int> bufferInputIndices,
                Dictionary<BufferVar, TIR.Buffer> abiBufferMemo,
                HashSet<int> storedOutputIndices,
                DistributedType?[] outputDistributedTypes,
                Dictionary<int, int> outputAliases,
                HashSet<PrimFunction> activePrimFunctionCalls,
                HashSet<string> activeDeviceFunctionNames,
                Dictionary<PrimFunction, DeviceFunctionDefinition>? deviceFunctionDefinitions = null,
                Dictionary<string, DeviceFunctionDefinition>? deviceFunctionDefinitionsByName = null)
            {
                InputNames = inputNames;
                KVCacheFieldInputs = kvCacheFieldInputs;
                RuntimeScalarNames = runtimeScalarNames;
                AbiViewStrideArgNames = abiViewStrideArgNames;
                BufferInputIndices = bufferInputIndices;
                AbiBufferMemo = abiBufferMemo;
                StoredOutputIndices = storedOutputIndices;
                OutputDistributedTypes = outputDistributedTypes;
                OutputAliases = outputAliases;
                ActivePrimFunctionCalls = activePrimFunctionCalls;
                ActiveDeviceFunctionNames = activeDeviceFunctionNames;
                DeviceFunctionDefinitions = deviceFunctionDefinitions ?? new Dictionary<PrimFunction, DeviceFunctionDefinition>(ReferenceEqualityComparer.Instance);
                DeviceFunctionDefinitionsByName = deviceFunctionDefinitionsByName ?? new Dictionary<string, DeviceFunctionDefinition>(StringComparer.Ordinal);
            }

            public List<string> InputNames { get; }

            public List<PyNTTKVCacheFieldInputMetadata> KVCacheFieldInputs { get; }

            public SortedSet<string> RuntimeScalarNames { get; }

            public SortedSet<string> AbiViewStrideArgNames { get; }

            public Dictionary<TIR.Buffer, int> BufferInputIndices { get; }

            public Dictionary<BufferVar, TIR.Buffer> AbiBufferMemo { get; }

            public HashSet<int> StoredOutputIndices { get; }

            public DistributedType?[] OutputDistributedTypes { get; }

            public Dictionary<int, int> OutputAliases { get; }

            public HashSet<PrimFunction> ActivePrimFunctionCalls { get; }

            public HashSet<string> ActiveDeviceFunctionNames { get; }

            public Dictionary<PrimFunction, DeviceFunctionDefinition> DeviceFunctionDefinitions { get; }

            public Dictionary<string, DeviceFunctionDefinition> DeviceFunctionDefinitionsByName { get; }
        }

        private sealed record BufferRef(
            string BaseName,
            string OffsetBytes,
            string PoolStrideBytes,
            string? IndexExpression,
            int[]? ShardCoordHierarchy,
            bool IsByteAddressed,
            string PoolScopeSize = "1",
            int AddressSpace = 1,
            SharedBufferAllocation? SharedAllocation = null,
            PyNTTDimExpression[]? LocalBaseCoordinates = null);

        private sealed record SharedBufferAllocation(
            string DescriptorName,
            long[] DescriptorShape,
            long[] LogicalShape,
            long[] LogicalStrides,
            int[] VectorLaneShape,
            long AvailableBytes,
            int ScalarElementSizeBytes,
            string TritonDType,
            string StorageEncoding);

        private sealed class ReductionState
        {
            public ReductionState(
                Call call,
                ReductionKernelKind kind,
                string[] names,
                int expectedUpdateCount)
            {
                Call = call;
                Kind = kind;
                Names = names;
                ExpectedUpdateCount = expectedUpdateCount;
            }

            public Call Call { get; }

            public ReductionKernelKind Kind { get; }

            public string[] Names { get; }

            public ReductionAccumulatorInitializer[]? Initializers { get; set; }

            public string? TemplatePath { get; set; }

            public object? FinalizeModel { get; set; }

            public string? FinalizeSignature { get; set; }

            public int ExpectedUpdateCount { get; }

            public int UpdateCount { get; set; }
        }

        private sealed record ReductionAccumulatorInitializer(string ValueExpression)
        {
            public static ReductionAccumulatorInitializer Register(string expression)
                => new(expression);
        }

        private sealed record OutputControlFlowState(
            HashSet<int> MayStore,
            HashSet<int> MustStore,
            DistributedType?[] OutputDistributedTypes,
            Dictionary<int, int> OutputAliases,
            Dictionary<int, string> FormalObjectOutputAliases);

        public enum DeviceFunctionFormalParameterKind
        {
            Tensor,
            Workspace,
            Scalar,
            Object,
        }

        public sealed record DeviceFunctionFormalParameter(
            int Index,
            IVar Parameter,
            DeviceFunctionFormalParameterKind Kind,
            string? BaseName,
            string? PoolStrideName,
            string? PoolScopeSizeName,
            string[] StrideNames,
            string[] DimensionNames,
            string[] GlobalOffsetNames,
            string? ScalarName,
            MemoryLocation? WorkspaceLocation)
        {
            public string? ObjectBaseName => ScalarName;
        }

        public sealed record DeviceFunctionFormalPlan(
            IReadOnlyDictionary<IVar, string> ParameterNames,
            IReadOnlyDictionary<IVar, string> TensorBaseNames,
            IReadOnlyDictionary<IVar, string> TensorPoolStrideNames,
            IReadOnlyDictionary<IVar, string> TensorPoolScopeSizeNames,
            IReadOnlyDictionary<IVar, PyNTTDimExpression[]> TensorDimensions,
            IReadOnlyDictionary<IVar, PyNTTDimExpression[]> TensorGlobalOffsets,
            IReadOnlyDictionary<IVar, int[][]> TensorSourceSplitAxes,
            IReadOnlyDictionary<string, string> DimParameterNames,
            IReadOnlyDictionary<IVar, string> ObjectBaseNames,
            IReadOnlySet<IVar> WorkspaceParameters,
            IReadOnlySet<string> ExtraParameters,
            IReadOnlyDictionary<string, string> ExtraPointerParameterTritonTypes,
            string DataBaseName,
            string ChipLocalDataBaseName,
            string BlockLocalDataBaseName,
            IReadOnlyList<DeviceFunctionFormalParameter> Parameters);

        public sealed record DeviceFunctionDefinition(
            string Name,
            DeviceFunctionBuildResult BuildResult,
            IReadOnlyList<DeviceFunctionFormalParameter> Parameters,
            IReadOnlyDictionary<IVar, int[][]> TensorSourceSplitAxes);
    }

    private static LaunchMetadata BuildLaunchMetadata(OutputInfo output, PyNTTTargetOptions targetOptions, Dictionary<string, object>? extraMeta = null)
    {
        var numel = Product(output.Shape);
        var meta = new Dictionary<string, object>
        {
            ["numel_expr"] = numel.PythonExpression,
            ["placement"] = "cuda",
        };
        if (extraMeta is not null)
        {
            foreach (var pair in extraMeta)
            {
                meta[pair.Key] = pair.Value;
            }
        }

        return new LaunchMetadata(
            meta,
            new(
                new()
                {
                    ["block_size"] = new TuningParameterMetadata("search_space", ElementwiseBlockSizeSearchSpace),
                }),
            BuildShardingMetadata(output, targetOptions),
            8,
            null);
    }

    private static ShardMetadata BuildShardingMetadata(OutputInfo output, PyNTTTargetOptions targetOptions)
    {
        if (output.DistributedType is { } distributedType)
        {
            return new ShardMetadata(
                "local_shard",
                string.IsNullOrWhiteSpace(distributedType.Placement.NormalizedHierarchyNames) ? "b" : distributedType.Placement.NormalizedHierarchyNames,
                GetTensorAxis(distributedType),
                "grid[0]",
                distributedType.Placement.Hierarchy.ToArray(),
                distributedType.Placement.NormalizedHierarchyLevels,
                ToPythonExpressions(output.Shape));
        }

        return new ShardMetadata(
            "local_shard",
            GetBlockPlacementAxis(targetOptions),
            0,
            "grid[0]",
            GetBlockHierarchy(targetOptions),
            GetBlockHierarchyLevels(targetOptions),
            ToPythonExpressions(output.Shape));
    }

    private static int GetTensorAxis(DistributedType distributedType)
    {
        for (var index = 0; index < distributedType.AxisPolicies.Count; index++)
        {
            if (distributedType.AxisPolicies[index] is SBPSplit)
            {
                return index;
            }
        }

        return 0;
    }

    private static string GetBlockPlacementAxis(PyNTTTargetOptions targetOptions)
    {
        var hierarchy = GetBlockHierarchy(targetOptions);
        var hierarchyNames = Placement.NormalizeAxisString(string.IsNullOrWhiteSpace(targetOptions.HierarchyNames)
            ? "b"
            : targetOptions.HierarchyNames);
        var hierarchyLevels = GetBlockHierarchyLevels(targetOptions);
        if (hierarchyNames.Length != hierarchy.Length)
        {
            hierarchyNames = new string('b', hierarchy.Length);
        }

        var blockAxes = hierarchyNames.Zip(hierarchyLevels).Where(pair => pair.Second == 'b').Select(pair => pair.First);
        var placementAxis = string.Concat(blockAxes);
        return string.IsNullOrEmpty(placementAxis) ? "b" : placementAxis;
    }

    private static int[] GetBlockHierarchy(PyNTTTargetOptions targetOptions)
    {
        var hierarchies = targetOptions.Hierarchies;
        return hierarchies.Length == 0 ? new[] { 1 } : hierarchies[0];
    }

    private static string GetBlockHierarchyLevels(PyNTTTargetOptions targetOptions)
    {
        var hierarchy = GetBlockHierarchy(targetOptions);
        return Placement.NormalizeHierarchyLevels(targetOptions.HierarchyLevels, targetOptions.HierarchyNames, hierarchy.Length);
    }

    private static string BuildGeneratedTopKernelPython(GeneratedKernelMetadata kernel, string bodySource, string helperSource)
    {
        var inputs = kernel.Inputs.Select((_, index) => $"input{index}").ToArray();
        var outputs = kernel.Outputs.Select((_, index) => $"output{index}").ToArray();
        var workspaceParameters = new[] { "data", "rdata", "chip_local_rdata", "chip_local_data", "block_local_rdata", "block_local_data" };
        var runtimeShapeArgs = GetRuntimeShapeArgs(kernel);
        var gridBarrierParameters = kernel.Attrs.ContainsKey("requires_grid_barrier")
            ? new[] { "pyntt_grid_mesh: tl.constexpr" }
            : Array.Empty<string>();
        var parameters = string.Join(", ", inputs.Concat(outputs).Concat(workspaceParameters).Concat(runtimeShapeArgs).Concat(gridBarrierParameters).Concat(new[] { "numel", "block_size: tl.constexpr" }));
        if (kernel.Attrs.TryGetValue("abi_view_stride_args", out var abiStrideArgsValue))
        {
            var abiStrideArgs = abiStrideArgsValue switch
            {
                string[] array => array,
                IEnumerable<string> enumerable => enumerable.ToArray(),
                _ => throw new NotSupportedException($"PyNTT kernel {kernel.Name} has unsupported abi_view_stride_args metadata type {abiStrideArgsValue.GetType().Name}."),
            };
            parameters = string.Join(", ", inputs.Concat(outputs).Concat(abiStrideArgs).Concat(workspaceParameters).Concat(runtimeShapeArgs).Concat(gridBarrierParameters).Concat(new[] { "numel", "block_size: tl.constexpr" }));
        }

        var call = IndentGeneratedCall(bodySource);

        var topKernelSource = $$"""
            @triton.jit
            def {{kernel.Name}}({{parameters}}):
                {{call}}
            """;
        return string.IsNullOrWhiteSpace(helperSource)
            ? topKernelSource
            : $"{helperSource.TrimEnd()}{Environment.NewLine}{Environment.NewLine}{topKernelSource}";
    }

    private static string IndentGeneratedCall(string call)
    {
        return call.Replace(Environment.NewLine, Environment.NewLine + "    ", StringComparison.Ordinal);
    }

    private static string GetTensorName(BaseExpr expr, IReadOnlyDictionary<IVar, string> parameterNames)
    {
        while (expr is Call call && call.Target is Boxing)
        {
            expr = call.Arguments[0];
        }

        if (expr is TIR.Buffer buffer
            && IsAbiBuffer(buffer)
            && buffer.MemSpan.Buffer.Start is IVar bufferParameter
            && parameterNames.TryGetValue(bufferParameter, out var bufferParameterName))
        {
            return bufferParameterName;
        }

        if (expr is IVar parameter && parameterNames.TryGetValue(parameter, out var name))
        {
            return name;
        }

        throw new NotSupportedException($"PyNTT M3 elementwise kernels only support direct function parameter operands, got {expr.GetType().Name}.");
    }

    private static bool IsAbiBuffer(TIR.Buffer buffer)
        => buffer.MemSpan.Buffer.Location is MemoryLocation.Input or MemoryLocation.Output;

    private static bool UsesBackingTensorLogicalLayout(TIR.Buffer buffer, IVar backingParameter)
    {
        var backingTensorType = backingParameter.CheckedType switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => null,
        };
        if (backingTensorType?.Shape is not RankedShape backingShape ||
            buffer.Rank != backingShape.Rank ||
            !buffer.ElemType.Equals(backingTensorType.DType) ||
            !buffer.Dimensions.SequenceEqual(backingShape.Dimensions))
        {
            return false;
        }

        return Equals(buffer.DistributedType, backingParameter.CheckedType as DistributedType);
    }

    private static DistributedType? GetDistributedType(BaseExpr expr)
    {
        return expr switch
        {
            Nncase.TIR.Buffer buffer => buffer.DistributedType,
            _ => expr.CheckedType as DistributedType,
        };
    }

    private static PyNTTDimExpression[] GetTensorShape(BaseExpr expr, string name)
    {
        var tensorType = GetTensorType(expr.CheckedType, name);
        return GetRankedShape(tensorType, name).Dimensions.ToArray()
            .Select(dimension => new PyNTTDimExpressionEmitter().Emit(dimension))
            .ToArray();
    }

    private static int? GetShardAxis(BaseExpr expr)
    {
        return GetDistributedType(expr) is { } distributedType ? GetShardAxis(distributedType) : null;
    }

    private static int? GetShardAxis(DistributedType distributedType)
    {
        for (var index = 0; index < distributedType.AxisPolicies.Count; index++)
        {
            if (distributedType.AxisPolicies[index] is SBPSplit)
            {
                return index;
            }
        }

        return null;
    }

    private static long GetBufferOffsetBytes(TIR.Buffer buffer)
    {
        var byteOffset = GetFixedDimension(buffer.MemSpan.Buffer.Start, $"{buffer.MemSpan.Buffer.Location} physical buffer offset") +
            GetFixedDimension(buffer.MemSpan.Start, $"{buffer.MemSpan.Buffer.Location} memspan offset");
        return byteOffset;
    }

    private static long GetPoolSizeBytes(IReadOnlyDictionary<Const, ValueRange<ulong>> ranges)
    {
        return ranges.Count == 0 ? 0L : checked((long)ranges.Values.Max(range => range.Max));
    }

    private static long GetFixedDimension(BaseExpr expr, string name)
    {
        try
        {
            return expr switch
            {
                None => 0,
                DimConst dimConst => dimConst.Value,
                Dimension dimension when dimension.IsFixed => dimension.FixedValue,
                TensorConst tensorConst => ReadScalarInt64(tensorConst.Value, $"PyNTT fixed {name}"),
                _ => expr.Evaluate().AsTensor().ToScalar<long>(),
            };
        }
        catch (Exception ex) when (ex is InvalidCastException or NotSupportedException)
        {
            throw new NotSupportedException($"PyNTT requires fixed {name}, got {expr}.", ex);
        }
    }

    private static long ReadScalarInt64(Tensor tensor, string context)
    {
        if (!tensor.Shape.IsScalar)
        {
            throw new NotSupportedException($"{context} expects a scalar tensor, got shape {tensor.Shape}.");
        }

        return tensor[Array.Empty<long>()] switch
        {
            sbyte value => value,
            byte value => value,
            short value => value,
            ushort value => value,
            int value => value,
            uint value => value,
            long value => value,
            ulong value => checked((long)value),
            bool value => value ? 1L : 0L,
            var value when TryReadPointerValue(value, out var pointerValue) => checked((long)pointerValue),
            var value => throw new NotSupportedException($"{context} expects an integral scalar tensor, got {value?.GetType().Name ?? "null"}."),
        };
    }

    private static bool TryReadPointerValue(object? value, out ulong pointerValue)
    {
        pointerValue = 0UL;
        if (value is null)
        {
            return false;
        }

        var type = value.GetType();
        if (!type.IsGenericType || type.GetGenericTypeDefinition() != typeof(Pointer<>))
        {
            return false;
        }

        pointerValue = (ulong)type.GetProperty(nameof(Pointer<byte>.Value))!.GetValue(value)!;
        return true;
    }

    private static string GetOpName(Enum op)
    {
        return op switch
        {
            UnaryOp.Abs => "abs",
            UnaryOp.Acos => "acos",
            UnaryOp.Acosh => "acosh",
            UnaryOp.Asin => "asin",
            UnaryOp.Asinh => "asinh",
            UnaryOp.Ceil => "ceil",
            UnaryOp.Cos => "cos",
            UnaryOp.Cosh => "cosh",
            UnaryOp.Erf => "erf",
            UnaryOp.Exp => "exp",
            UnaryOp.Floor => "floor",
            UnaryOp.Log => "log",
            UnaryOp.Neg => "neg",
            UnaryOp.Round => "round",
            UnaryOp.Rsqrt => "rsqrt",
            UnaryOp.Sin => "sin",
            UnaryOp.Sinh => "sinh",
            UnaryOp.Sqrt => "sqrt",
            UnaryOp.Square => "square",
            UnaryOp.Tanh => "tanh",
            UnaryOp.LogicalNot => "logical_not",
            UnaryOp.Sign => "sign",
            BinaryOp.Add => "add",
            BinaryOp.Sub => "sub",
            BinaryOp.Mul => "mul",
            BinaryOp.Div => "div",
            BinaryOp.Mod => "mod",
            BinaryOp.Min => "min",
            BinaryOp.Max => "max",
            _ => throw new NotSupportedException($"Unsupported PyNTT elementwise op: {op}."),
        };
    }

    private static string GetUnaryExpression(UnaryOp op)
    {
        return op switch
        {
            UnaryOp.Abs => "tl.abs(value0)",
            UnaryOp.Acos => "libdevice.acos(value0_f32)",
            UnaryOp.Acosh => "tl.log(value0_f32 + tl.sqrt(value0_f32 * value0_f32 - 1.0))",
            UnaryOp.Asin => "libdevice.asin(value0_f32)",
            UnaryOp.Asinh => "tl.log(value0_f32 + tl.sqrt(value0_f32 * value0_f32 + 1.0))",
            UnaryOp.Ceil => "tl.ceil(value0)",
            UnaryOp.Cos => "tl.cos(value0_f32)",
            UnaryOp.Cosh => "(tl.exp(value0_f32) + tl.exp(-value0_f32)) * 0.5",
            UnaryOp.Erf => "tl.erf(value0_f32)",
            UnaryOp.Exp => "tl.exp(value0_f32)",
            UnaryOp.Floor => "tl.floor(value0)",
            UnaryOp.Log => "tl.log(value0_f32)",
            UnaryOp.Neg => "-value0",
            UnaryOp.Round => "libdevice.round(value0)",
            UnaryOp.Rsqrt => "tl.rsqrt(value0_f32)",
            UnaryOp.Sin => "tl.sin(value0_f32)",
            UnaryOp.Sinh => "(tl.exp(value0_f32) - tl.exp(-value0_f32)) * 0.5",
            UnaryOp.Sqrt => "tl.sqrt(value0_f32)",
            UnaryOp.Square => "value0 * value0",
            UnaryOp.Tanh => "(tl.exp(value0_f32 + value0_f32) - 1.0) / (tl.exp(value0_f32 + value0_f32) + 1.0)",
            UnaryOp.LogicalNot => "value0 == 0",
            UnaryOp.Sign => "tl.where(value0 > 0, 1.0, tl.where(value0 < 0, -1.0, 0.0))",
            _ => throw new NotSupportedException($"Unsupported PyNTT unary template op: {op}."),
        };
    }

    private static string GetGluTypeName(GluType gluType)
    {
        return gluType switch
        {
            GluType.SwiGLU => "swiglu",
            _ => throw new NotSupportedException($"Unsupported PyNTT MatMulGlu type: {gluType}."),
        };
    }

    private static string GetBinaryExpression(BinaryOp op)
    {
        return op switch
        {
            BinaryOp.Add => "value0 + value1",
            BinaryOp.Sub => "value0 - value1",
            BinaryOp.Mul => "value0 * value1",
            BinaryOp.Div => "value0 / value1",
            BinaryOp.Mod => "value0 % value1",
            BinaryOp.Min => "tl.minimum(value0, value1)",
            BinaryOp.Max => "tl.maximum(value0, value1)",
            _ => throw new NotSupportedException($"Unsupported PyNTT binary template op: {op}."),
        };
    }

    private static string GetCompareOpName(CompareOp op)
    {
        return op switch
        {
            CompareOp.Equal => "equal",
            CompareOp.NotEqual => "not_equal",
            CompareOp.LowerThan => "less",
            CompareOp.LowerOrEqual => "less_or_equal",
            CompareOp.GreaterThan => "greater",
            CompareOp.GreaterOrEqual => "greater_or_equal",
            _ => throw new NotSupportedException($"Unsupported PyNTT compare op: {op}."),
        };
    }

    private static string GetCompareExpression(CompareOp op)
    {
        return op switch
        {
            CompareOp.Equal => "value0 == value1",
            CompareOp.NotEqual => "value0 != value1",
            CompareOp.LowerThan => "value0 < value1",
            CompareOp.LowerOrEqual => "value0 <= value1",
            CompareOp.GreaterThan => "value0 > value1",
            CompareOp.GreaterOrEqual => "value0 >= value1",
            _ => throw new NotSupportedException($"Unsupported PyNTT compare template op: {op}."),
        };
    }

    private static string GetCompareOperator(CompareOp op)
    {
        return op switch
        {
            CompareOp.Equal => "==",
            CompareOp.NotEqual => "!=",
            CompareOp.LowerThan => "<",
            CompareOp.LowerOrEqual => "<=",
            CompareOp.GreaterThan => ">",
            CompareOp.GreaterOrEqual => ">=",
            _ => throw new NotSupportedException($"Unsupported PyNTT compare op: {op}."),
        };
    }

    private static string BuildScalarBinaryExpression(BinaryOp op, string lhs, string rhs)
    {
        return op switch
        {
            BinaryOp.Add => $"({lhs} + {rhs})",
            BinaryOp.Sub => $"({lhs} - {rhs})",
            BinaryOp.Mul => $"({lhs} * {rhs})",
            BinaryOp.Div => $"({lhs} / {rhs})",
            BinaryOp.Mod => $"({lhs} % {rhs})",
            BinaryOp.Min => $"tl.minimum({lhs}, {rhs})",
            BinaryOp.Max => $"tl.maximum({lhs}, {rhs})",
            BinaryOp.Pow => $"(({lhs}) ** ({rhs}))",
            BinaryOp.BitwiseAnd => $"({lhs} & {rhs})",
            BinaryOp.BitwiseOr => $"({lhs} | {rhs})",
            BinaryOp.BitwiseXor => $"({lhs} ^ {rhs})",
            BinaryOp.LogicalAnd => $"({lhs} and {rhs})",
            BinaryOp.LogicalOr => $"({lhs} or {rhs})",
            BinaryOp.LogicalXor => $"(({lhs}) != ({rhs}))",
            BinaryOp.LeftShift => $"({lhs} << {rhs})",
            BinaryOp.RightShift => $"({lhs} >> {rhs})",
            BinaryOp.FloorDiv => $"(({lhs}) // ({rhs}))",
            BinaryOp.CeilDiv => $"((({lhs}) + ({rhs}) - 1) // ({rhs}))",
            _ => throw new NotSupportedException($"Unsupported PyNTT scalar binary op: {op}."),
        };
    }

    private static string FormatScalarConst(TensorConst tensorConst)
    {
        if (!tensorConst.Value.Shape.IsScalar)
        {
            throw new NotSupportedException("PyNTT scalar expression only supports scalar TensorConst values.");
        }

        var value = tensorConst.Value[Array.Empty<long>()];
        return value switch
        {
            bool boolean => boolean ? "True" : "False",
            IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
            _ => value?.ToString() ?? "None",
        };
    }

    private static string GetClampExpression(float min, float max)
    {
        return (float.IsNegativeInfinity(min), float.IsPositiveInfinity(max)) switch
        {
            (true, true) => "value0",
            (true, false) => $"tl.minimum(value0, {FormatFloat(max)})",
            (false, true) => $"tl.maximum(value0, {FormatFloat(min)})",
            (false, false) => $"tl.minimum(tl.maximum(value0, {FormatFloat(min)}), {FormatFloat(max)})",
        };
    }

    private static string FormatFloat(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    private static string GetCastModeName(CastMode castMode)
    {
        return castMode switch
        {
            CastMode.KDefault => "kdefault",
            CastMode.Exact => "exact",
            CastMode.CheckOverflow => "check_overflow",
            CastMode.Reinterpret => "reinterpret",
            _ => throw new NotSupportedException($"Unsupported PyNTT cast mode: {castMode}."),
        };
    }

    private static string GetCastExpression(CastMode castMode, DataType outputDataType)
    {
        if (GetScalarDataType(outputDataType) == DataTypes.Boolean)
        {
            return "value0 != 0";
        }

        var outputTritonDType = GetScalarTritonDType(outputDataType);
        return castMode switch
        {
            CastMode.KDefault or CastMode.Exact or CastMode.CheckOverflow => $"value0.to({outputTritonDType})",
            CastMode.Reinterpret => $"value0.to({outputTritonDType}, bitcast=True)",
            _ => throw new NotSupportedException($"Unsupported PyNTT cast mode: {castMode}."),
        };
    }

    private static string GetReduceOpName(ReduceOp op)
    {
        return op switch
        {
            ReduceOp.Sum => "sum",
            ReduceOp.Mean => "mean",
            ReduceOp.Max => "max",
            ReduceOp.Min => "min",
            _ => throw new NotSupportedException($"Unsupported PyNTT reduce op: {op}."),
        };
    }

    private static string GetReduceInitValue(ReduceOp op)
    {
        return op switch
        {
            ReduceOp.Sum => "0.0",
            ReduceOp.Mean => "0.0",
            ReduceOp.Max => "-float(\"inf\")",
            ReduceOp.Min => "float(\"inf\")",
            _ => throw new NotSupportedException($"Unsupported PyNTT reduce op: {op}."),
        };
    }

    private static string GetReduceInitValue(ReduceOp op, DataType dataType)
    {
        if (DataTypes.IsFloat(dataType))
        {
            return GetReduceInitValue(op);
        }

        if (!DataTypes.IsIntegral(dataType))
        {
            throw new NotSupportedException($"PyNTT Reduce does not support accumulator dtype {dataType}.");
        }

        return op switch
        {
            ReduceOp.Sum or ReduceOp.Mean => "0",
            ReduceOp.Max when dataType == DataTypes.Int8 => sbyte.MinValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Max when dataType == DataTypes.Int16 => short.MinValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Max when dataType == DataTypes.Int32 => int.MinValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Max when dataType == DataTypes.Int64 => long.MinValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Max when dataType is UInt8Type or UInt16Type or UInt32Type or UInt64Type => "0",
            ReduceOp.Min when dataType == DataTypes.Int8 => sbyte.MaxValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Min when dataType == DataTypes.Int16 => short.MaxValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Min when dataType == DataTypes.Int32 => int.MaxValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Min when dataType == DataTypes.Int64 => long.MaxValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Min when dataType == DataTypes.UInt8 => byte.MaxValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Min when dataType == DataTypes.UInt16 => ushort.MaxValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Min when dataType == DataTypes.UInt32 => uint.MaxValue.ToString(CultureInfo.InvariantCulture),
            ReduceOp.Min when dataType == DataTypes.UInt64 => ulong.MaxValue.ToString(CultureInfo.InvariantCulture),
            _ => throw new NotSupportedException($"Unsupported PyNTT {op} accumulator dtype {dataType}."),
        };
    }

    private static string GetReduceUpdateExpression(ReduceOp op)
    {
        return op switch
        {
            ReduceOp.Sum => "acc + value0",
            ReduceOp.Mean => "acc + value0",
            ReduceOp.Max => "tl.maximum(acc, value0)",
            ReduceOp.Min => "tl.minimum(acc, value0)",
            _ => throw new NotSupportedException($"Unsupported PyNTT reduce op: {op}."),
        };
    }

    private static string GetReduceFinalizeExpression(ReduceOp op, PyNTTDimExpression reduceElementCount)
    {
        return op switch
        {
            ReduceOp.Sum or ReduceOp.Max or ReduceOp.Min => "acc",
            ReduceOp.Mean => $"acc / (({reduceElementCount.TritonExpression}) + lane * 0).to(tl.float32)",
            _ => throw new NotSupportedException($"Unsupported PyNTT reduce op: {op}."),
        };
    }

    private static bool GetScalarBool(BaseExpr expr, string name)
    {
        if (expr is None)
        {
            return false;
        }

        try
        {
            return expr.Evaluate().AsTensor().ToScalar<bool>();
        }
        catch (Exception ex) when (ex is InvalidCastException or NotSupportedException)
        {
            throw new NotSupportedException($"PyNTT requires a bool scalar for {name}.", ex);
        }
    }

    private static float GetScalarFloat(BaseExpr expr, string name, float defaultValue)
    {
        if (expr is None)
        {
            return defaultValue;
        }

        try
        {
            return expr.Evaluate().AsTensor().ToScalar<float>();
        }
        catch (Exception ex) when (ex is InvalidCastException or NotSupportedException)
        {
            throw new NotSupportedException($"PyNTT requires a float32 scalar for {name}.", ex);
        }
    }

    private static int NormalizeAxis(int axis, int rank, string context)
    {
        var normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank)
        {
            throw new NotSupportedException($"{context} axis {axis} is out of range for rank {rank}.");
        }

        return normalized;
    }

    private static int[] NormalizeAxes(IReadOnlyList<int> axes, int rank, string context)
    {
        if (axes.Count == 0)
        {
            throw new NotSupportedException($"{context} requires at least one reduction axis.");
        }

        var normalizedAxes = axes
            .Select(axis => NormalizeAxis(axis, rank, context))
            .OrderBy(axis => axis)
            .ToArray();
        for (var i = 1; i < normalizedAxes.Length; i++)
        {
            if (normalizedAxes[i] == normalizedAxes[i - 1])
            {
                throw new NotSupportedException($"{context} contains duplicated axis {normalizedAxes[i]}.");
            }
        }

        return normalizedAxes;
    }

    private static bool SameDim(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (lhs.FixedValue.HasValue && rhs.FixedValue.HasValue)
        {
            return lhs.FixedValue.Value == rhs.FixedValue.Value;
        }

        if (lhs.PythonExpression == rhs.PythonExpression ||
            lhs.TritonExpression == rhs.TritonExpression)
        {
            return true;
        }

        if (lhs.MinValue.HasValue &&
            rhs.MaxValue.HasValue &&
            lhs.MinValue.Value > rhs.MaxValue.Value)
        {
            return false;
        }

        if (rhs.MinValue.HasValue &&
            lhs.MaxValue.HasValue &&
            rhs.MinValue.Value > lhs.MaxValue.Value)
        {
            return false;
        }

        if (lhs.FixedValue.HasValue && rhs.MinValue.HasValue && rhs.MaxValue.HasValue)
        {
            return lhs.FixedValue.Value >= rhs.MinValue.Value && lhs.FixedValue.Value <= rhs.MaxValue.Value;
        }

        if (rhs.FixedValue.HasValue && lhs.MinValue.HasValue && lhs.MaxValue.HasValue)
        {
            return rhs.FixedValue.Value >= lhs.MinValue.Value && rhs.FixedValue.Value <= lhs.MaxValue.Value;
        }

        return !lhs.FixedValue.HasValue && !rhs.FixedValue.HasValue;
    }

    private static bool IsBoundedExtent(PyNTTDimExpression dimension, long expected)
    {
        if (dimension.FixedValue.HasValue)
        {
            return dimension.FixedValue.Value == expected;
        }

        return dimension.MinValue is >= 0 &&
            dimension.MaxValue.HasValue &&
            dimension.MaxValue.Value <= expected;
    }

    private static bool EquivalentDim(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
        => lhs.IsEquivalentTo(rhs);

    private static PyNTTDimExpression WithAffineEquivalence(
        PyNTTDimExpression expression,
        PyNTTDimExpression lhs,
        PyNTTDimExpression rhs,
        bool subtract)
    {
        if (lhs.Equivalence is not { } lhsEquivalence ||
            rhs.Equivalence is not { } rhsEquivalence)
        {
            return expression.EnsureEquivalence();
        }

        var equivalence = subtract
            ? PyNTTDimEquivalence.TrySubtract(lhsEquivalence, rhsEquivalence)
            : PyNTTDimEquivalence.TryAdd(lhsEquivalence, rhsEquivalence);
        return equivalence is null
            ? expression.EnsureEquivalence()
            : expression with
            {
                Equivalence = equivalence,
            };
    }

    private static bool CanFitPaddedDim(PyNTTDimExpression actual, PyNTTDimExpression padded)
    {
        if (actual.FixedValue.HasValue && padded.FixedValue.HasValue)
        {
            return actual.FixedValue.Value <= padded.FixedValue.Value;
        }

        return SameDim(actual, padded);
    }

    private static string ShapeText(IEnumerable<PyNTTDimExpression> shape)
        => string.Join(",", shape.Select(dim => dim.PythonExpression));

    private static long RequireFixedDim(PyNTTDimExpression dimension, string context)
        => dimension.FixedValue ?? throw new NotSupportedException($"{context} must be fixed for the current PyNTT codegen path, got {dimension.PythonExpression}.");

    private static PyNTTDimExpression ToDim(long value)
    {
        var text = value.ToString(CultureInfo.InvariantCulture);
        return new(text, text, value)
        {
            Equivalence = PyNTTDimEquivalence.FromConstant(value),
        };
    }

    private static PyNTTDimExpression AddDims(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (lhs.FixedValue == 0)
        {
            return rhs;
        }

        if (rhs.FixedValue == 0)
        {
            return lhs;
        }

        long? fixedValue = lhs.FixedValue.HasValue && rhs.FixedValue.HasValue
            ? checked(lhs.FixedValue.Value + rhs.FixedValue.Value)
            : null;
        long? rangeMin = lhs.MinValue.HasValue && rhs.MinValue.HasValue
            ? checked(lhs.MinValue.Value + rhs.MinValue.Value)
            : null;
        long? rangeMax = lhs.MaxValue.HasValue && rhs.MaxValue.HasValue
            ? checked(lhs.MaxValue.Value + rhs.MaxValue.Value)
            : null;
        var expression = new PyNTTDimExpression(
            $"({lhs.PythonExpression} + {rhs.PythonExpression})",
            $"({lhs.TritonExpression} + {rhs.TritonExpression})",
            fixedValue,
            rangeMin,
            rangeMax);
        return WithAffineEquivalence(expression, lhs, rhs, subtract: false);
    }

    private static PyNTTDimExpression SubtractDims(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (rhs.FixedValue == 0)
        {
            return lhs;
        }

        long? fixedValue = lhs.FixedValue.HasValue && rhs.FixedValue.HasValue
            ? checked(lhs.FixedValue.Value - rhs.FixedValue.Value)
            : null;
        long? rangeMin = lhs.MinValue.HasValue && rhs.MaxValue.HasValue
            ? checked(lhs.MinValue.Value - rhs.MaxValue.Value)
            : null;
        long? rangeMax = lhs.MaxValue.HasValue && rhs.MinValue.HasValue
            ? checked(lhs.MaxValue.Value - rhs.MinValue.Value)
            : null;
        var expression = new PyNTTDimExpression(
            $"({lhs.PythonExpression} - {rhs.PythonExpression})",
            $"({lhs.TritonExpression} - {rhs.TritonExpression})",
            fixedValue,
            rangeMin,
            rangeMax);
        return WithAffineEquivalence(expression, lhs, rhs, subtract: true);
    }

    private static PyNTTDimExpression MinDims(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (EquivalentDim(lhs, rhs))
        {
            return lhs;
        }

        if (lhs.FixedValue.HasValue && rhs.FixedValue.HasValue)
        {
            return ToDim(Math.Min(lhs.FixedValue.Value, rhs.FixedValue.Value));
        }

        var minValues = new[] { lhs.MinValue, rhs.MinValue }.OfType<long>().ToArray();
        var maxValues = new[] { lhs.MaxValue, rhs.MaxValue }.OfType<long>().ToArray();
        long? rangeMin = minValues.Length == 2 ? minValues.Min() : null;
        long? rangeMax = maxValues.Length > 0 ? maxValues.Min() : null;
        return new PyNTTDimExpression(
            $"min({lhs.PythonExpression}, {rhs.PythonExpression})",
            $"tl.minimum({lhs.TritonExpression}, {rhs.TritonExpression})",
            null,
            rangeMin,
            rangeMax)
            .EnsureEquivalence();
    }

    private static PyNTTDimExpression MaxDims(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (EquivalentDim(lhs, rhs))
        {
            return lhs;
        }

        if (lhs.FixedValue.HasValue && rhs.FixedValue.HasValue)
        {
            return ToDim(Math.Max(lhs.FixedValue.Value, rhs.FixedValue.Value));
        }

        var minValues = new[] { lhs.MinValue, rhs.MinValue }.OfType<long>().ToArray();
        var maxValues = new[] { lhs.MaxValue, rhs.MaxValue }.OfType<long>().ToArray();
        long? rangeMin = minValues.Length > 0 ? minValues.Max() : null;
        long? rangeMax = maxValues.Length == 2 ? maxValues.Max() : null;
        return new PyNTTDimExpression(
            $"max({lhs.PythonExpression}, {rhs.PythonExpression})",
            $"tl.maximum({lhs.TritonExpression}, {rhs.TritonExpression})",
            null,
            rangeMin,
            rangeMax)
            .EnsureEquivalence();
    }

    private static PyNTTDimExpression MultiplyDim(PyNTTDimExpression lhs, long rhs)
    {
        if (rhs == 0 || lhs.FixedValue == 0)
        {
            return PyNTTDimExpression.Zero;
        }

        if (rhs == 1)
        {
            return lhs;
        }

        long? fixedValue = lhs.FixedValue.HasValue ? checked(lhs.FixedValue.Value * rhs) : null;
        long? rangeMin = lhs.MinValue.HasValue ? checked(lhs.MinValue.Value * rhs) : null;
        long? rangeMax = lhs.MaxValue.HasValue ? checked(lhs.MaxValue.Value * rhs) : null;
        if (rhs < 0)
        {
            (rangeMin, rangeMax) = (rangeMax, rangeMin);
        }

        var expression = new PyNTTDimExpression(
            $"({lhs.PythonExpression} * {rhs.ToString(CultureInfo.InvariantCulture)})",
            $"({lhs.TritonExpression} * {rhs.ToString(CultureInfo.InvariantCulture)})",
            fixedValue,
            rangeMin,
            rangeMax);
        return lhs.Equivalence is { } equivalence
            ? expression with { Equivalence = PyNTTDimEquivalence.Scale(equivalence, rhs) }
            : expression.EnsureEquivalence();
    }

    private static int GetSingleVectorLaneCount(DataType dataType, string context)
    {
        var lanes = GetVectorLanes(dataType);
        return lanes.Length switch
        {
            0 => 1,
            1 => lanes[0],
            _ => throw new NotSupportedException($"{context} expects at most one vector lane dimension, got [{string.Join(",", lanes)}]."),
        };
    }

    private static int GetVectorLaneElementCount(DataType dataType)
    {
        var lanes = GetVectorLanes(dataType);
        return lanes.Length == 0 ? 1 : lanes.Aggregate(1, (acc, lane) => checked(acc * lane));
    }

    private static void ValidateMatchingVectorLanes(string context, DataType lhs, DataType rhs)
    {
        var lhsLanes = GetVectorLanes(lhs);
        var rhsLanes = GetVectorLanes(rhs);
        if (!lhsLanes.SequenceEqual(rhsLanes))
        {
            throw new NotSupportedException($"{context} expects matching vector lanes, got lhs=[{string.Join(",", lhsLanes)}], rhs=[{string.Join(",", rhsLanes)}].");
        }
    }

    private static void ValidateScalarOrMatchingVectorLanes(string context, DataType operand, DataType output)
    {
        var operandLanes = GetVectorLanes(operand);
        var outputLanes = GetVectorLanes(output);
        if (operandLanes.Length != 0 && !operandLanes.SequenceEqual(outputLanes))
        {
            throw new NotSupportedException($"{context} expects scalar lanes or lanes matching output, got operand=[{string.Join(",", operandLanes)}], output=[{string.Join(",", outputLanes)}].");
        }
    }

    private static PyNTTDimExpression[] GetLogicalVectorShape(IReadOnlyList<PyNTTDimExpression> physicalShape, int laneCount)
    {
        if (laneCount == 1)
        {
            return physicalShape.ToArray();
        }

        if (physicalShape.Count == 0)
        {
            throw new NotSupportedException("PyNTT vector scalar buffers are not supported by the current Triton templates.");
        }

        var logicalShape = physicalShape.ToArray();
        logicalShape[^1] = MultiplyDim(logicalShape[^1], laneCount);
        return logicalShape;
    }

    private static PyNTTDimExpression[] GetLogicalVectorShape(IReadOnlyList<PyNTTDimExpression> physicalShape, int vectorizedAxis, int laneCount)
    {
        if (laneCount == 1)
        {
            return physicalShape.ToArray();
        }

        if (physicalShape.Count == 0)
        {
            throw new NotSupportedException("PyNTT vector scalar buffers are not supported by the current Triton templates.");
        }

        var axis = NormalizeAxis(vectorizedAxis, physicalShape.Count, "PyNTT vectorized axis");
        var logicalShape = physicalShape.ToArray();
        logicalShape[axis] = MultiplyDim(logicalShape[axis], laneCount);
        return logicalShape;
    }

    private static PyNTTDimExpression CeilDivDim(PyNTTDimExpression lhs, long rhs)
    {
        if (rhs <= 0)
        {
            throw new NotSupportedException($"PyNTT dimension ceil-div requires positive divisor, got {rhs}.");
        }

        if (rhs == 1)
        {
            return lhs;
        }

        long? fixedValue = lhs.FixedValue.HasValue ? checked((lhs.FixedValue.Value + rhs - 1) / rhs) : null;
        long? rangeMin = lhs.MinValue.HasValue ? checked((lhs.MinValue.Value + rhs - 1) / rhs) : null;
        long? rangeMax = lhs.MaxValue.HasValue ? checked((lhs.MaxValue.Value + rhs - 1) / rhs) : null;
        var rhsText = rhs.ToString(CultureInfo.InvariantCulture);
        return new PyNTTDimExpression(
            $"(({lhs.PythonExpression} + {rhsText} - 1) // ({rhsText}))",
            $"(({lhs.TritonExpression} + {rhsText} - 1) // ({rhsText}))",
            fixedValue,
            rangeMin,
            rangeMax)
            .EnsureEquivalence();
    }

    private static PyNTTDimExpression FloorDivDim(PyNTTDimExpression lhs, long rhs)
    {
        if (rhs <= 0)
        {
            throw new NotSupportedException($"PyNTT dimension floor-div requires positive divisor, got {rhs}.");
        }

        if (rhs == 1)
        {
            return lhs;
        }

        long? fixedValue = lhs.FixedValue.HasValue ? checked(lhs.FixedValue.Value / rhs) : null;
        long? rangeMin = lhs.MinValue.HasValue ? checked(lhs.MinValue.Value / rhs) : null;
        long? rangeMax = lhs.MaxValue.HasValue ? checked(lhs.MaxValue.Value / rhs) : null;
        var rhsText = rhs.ToString(CultureInfo.InvariantCulture);
        return new PyNTTDimExpression(
            $"(({lhs.PythonExpression}) // ({rhsText}))",
            $"(({lhs.TritonExpression}) // ({rhsText}))",
            fixedValue,
            rangeMin,
            rangeMax)
            .EnsureEquivalence();
    }

    private static void ValidateSameShape(string context, IReadOnlyList<PyNTTDimExpression> actual, IReadOnlyList<PyNTTDimExpression> expected)
    {
        if (actual.Count != expected.Count || actual.Zip(expected).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} requires matching shapes, got [{ShapeText(actual)}] and [{ShapeText(expected)}].");
        }
    }

    private static void ValidateMatchingFixedScalarElementCount(
        string context,
        IReadOnlyList<PyNTTDimExpression> inputShape,
        int inputLaneCount,
        IReadOnlyList<PyNTTDimExpression> outputShape,
        int outputLaneCount)
    {
        var inputCount = MultiplyDim(Product(inputShape), inputLaneCount);
        var outputCount = MultiplyDim(Product(outputShape), outputLaneCount);
        if (inputCount.FixedValue.HasValue &&
            outputCount.FixedValue.HasValue &&
            inputCount.FixedValue != outputCount.FixedValue)
        {
            throw new NotSupportedException($"{context} requires equal scalar element counts, got input={inputCount.FixedValue} and output={outputCount.FixedValue}.");
        }
    }

    private static void ValidatePackShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> outputShape, IReadOnlyList<int> axes, IReadOnlyList<int> lanes)
    {
        ValidateSameShape(context, outputShape, GetPackedShape(inputShape, axes, lanes));
    }

    private static void ValidateUnpackShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> outputShape, IReadOnlyList<int> axes, IReadOnlyList<int> lanes)
    {
        ValidateSameShape(context, outputShape, GetUnpackedShape(inputShape, axes, lanes));
    }

    private static PyNTTDimExpression[] GetPackedShape(IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<int> axes, IReadOnlyList<int> lanes)
        => inputShape
            .Select((dimension, axis) =>
            {
                var laneProduct = GetLayoutAxisLaneProduct(axes, lanes, axis);
                return laneProduct > 1 ? CeilDivDim(dimension, laneProduct) : dimension;
            })
            .ToArray();

    private static PyNTTDimExpression[] GetUnpackedShape(IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<int> axes, IReadOnlyList<int> lanes)
        => inputShape
            .Select((dimension, axis) =>
            {
                var laneProduct = GetLayoutAxisLaneProduct(axes, lanes, axis);
                return laneProduct > 1 ? MultiplyDim(dimension, laneProduct) : dimension;
            })
            .ToArray();

    private static int[] NormalizeLayoutAxes(IRArray<int> axes, int rank, string context)
    {
        var normalizedAxes = axes.IsDefaultOrEmpty
            ? Array.Empty<int>()
            : axes.ToArray().Select(axis => NormalizeAxis(axis, rank, context)).ToArray();

        return normalizedAxes;
    }

    private static int GetLayoutAxisLaneProduct(IReadOnlyList<int> axes, IReadOnlyList<int> lanes, int axis)
    {
        var product = 1;
        for (var i = 0; i < axes.Count; i++)
        {
            if (axes[i] == axis)
            {
                product = checked(product * lanes[i]);
            }
        }

        return product;
    }

    private static int[] GetLayoutLanes(IRArray<int> lanes, int axesCount, string context)
    {
        var laneArray = lanes.IsDefaultOrEmpty ? Array.Empty<int>() : lanes.ToArray();
        if (laneArray.Length != axesCount)
        {
            throw new NotSupportedException($"{context} expects lanes rank {axesCount}, got {laneArray.Length}.");
        }

        if (laneArray.Any(lane => lane <= 0))
        {
            throw new NotSupportedException($"{context} requires positive lanes, got [{string.Join(",", laneArray)}].");
        }

        return laneArray;
    }

    private static void ValidateLanePrefix(string context, IReadOnlyList<int> expected, IReadOnlyList<int> actual)
    {
        if (!expected.SequenceEqual(actual))
        {
            throw new NotSupportedException($"{context} mismatch, expected [{string.Join(",", expected)}], got [{string.Join(",", actual)}].");
        }
    }

    private static void ValidateRoPEShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> cosShape, IReadOnlyList<PyNTTDimExpression> sinShape, IReadOnlyList<PyNTTDimExpression> outputShape, int rotaryAxis, int laneCount, int sinCosVectorPackFactor)
    {
        ValidateSameShape(context, inputShape, outputShape);
        var sinCosOutputShape = outputShape.ToArray();
        if (sinCosVectorPackFactor > 1)
        {
            sinCosOutputShape[rotaryAxis] = CeilDivDim(sinCosOutputShape[rotaryAxis], sinCosVectorPackFactor);
        }

        ValidateBroadcastable($"{context} cos", cosShape, sinCosOutputShape);
        ValidateBroadcastable($"{context} sin", sinShape, sinCosOutputShape);
        if (outputShape.Count == 0)
        {
            throw new NotSupportedException($"{context} requires a non-scalar output, got [{ShapeText(outputShape)}].");
        }

        var rotaryDim = MultiplyDim(outputShape[rotaryAxis], laneCount);
        if (laneCount % 2 == 0)
        {
            return;
        }

        if (!rotaryDim.FixedValue.HasValue)
        {
            throw new NotSupportedException($"{context} cannot prove that dynamic rotary dimension {rotaryDim.PythonExpression} is even.");
        }

        if (rotaryDim.FixedValue.Value % 2 != 0)
        {
            throw new NotSupportedException($"{context} requires an even rotary dimension, got axis {rotaryAxis} with shape [{ShapeText(outputShape)}] and lanes {laneCount}.");
        }
    }

    private static int GetRoPESinCosVectorPackFactor(IReadOnlyList<int> inputLanes, IReadOnlyList<int> cosLanes, IReadOnlyList<int> sinLanes, IReadOnlyList<int> outputLanes)
    {
        if (!inputLanes.SequenceEqual(outputLanes))
        {
            throw new NotSupportedException($"PyNTT RoPE expects matching input/output vector lanes, got input=[{string.Join(",", inputLanes)}], output=[{string.Join(",", outputLanes)}].");
        }

        if (!cosLanes.SequenceEqual(sinLanes))
        {
            throw new NotSupportedException($"PyNTT RoPE expects matching cos/sin vector lanes, got cos=[{string.Join(",", cosLanes)}], sin=[{string.Join(",", sinLanes)}].");
        }

        if (cosLanes.Count == 0 && outputLanes.Count == 0)
        {
            return 1;
        }

        if (cosLanes.SequenceEqual(outputLanes))
        {
            return 1;
        }

        if (outputLanes.Count == 1 && cosLanes.Count == 2 && cosLanes[0] == 2 && cosLanes[1] == outputLanes[0])
        {
            return 2;
        }

        throw new NotSupportedException($"PyNTT RoPE expects sin/cos lanes to be scalar, [{string.Join(",", outputLanes)}], or [2,{string.Join(",", outputLanes)}], got [{string.Join(",", cosLanes)}].");
    }

    private static void ValidateLayerNormShape(string context, IReadOnlyList<PyNTTDimExpression> parameterShape, IReadOnlyList<PyNTTDimExpression> outputShape, int axis)
    {
        var expectedShape = outputShape.Skip(axis).ToArray();
        if (parameterShape.Count != expectedShape.Length || parameterShape.Zip(expectedShape).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} requires shape [{ShapeText(expectedShape)}], got [{ShapeText(parameterShape)}].");
        }
    }

    private static void ValidateNormStatsShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> statsShape, int axis, bool useMean)
    {
        var expectedRank = inputShape.Count + 1;
        if (statsShape.Count != expectedRank)
        {
            throw new NotSupportedException($"{context} stats rank must be {expectedRank}, got [{ShapeText(statsShape)}].");
        }

        var expectedComponents = useMean ? 2 : 1;
        if (!IsBoundedExtent(statsShape[0], expectedComponents))
        {
            throw new NotSupportedException($"{context} stats component dimension must be {expectedComponents}, got [{ShapeText(statsShape)}].");
        }

        for (var i = 0; i < inputShape.Count; i++)
        {
            if (i < axis)
            {
                if (!SameDim(inputShape[i], statsShape[i + 1]))
                {
                    throw new NotSupportedException($"{context} stats outer axis {i} must match input shape [{ShapeText(inputShape)}], got [{ShapeText(statsShape)}].");
                }
            }
            else if (!IsBoundedExtent(statsShape[i + 1], 1))
            {
                throw new NotSupportedException($"{context} stats normalized axis {i} must have extent 1, got [{ShapeText(statsShape)}].");
            }
        }
    }

    private static void ValidateBroadcastable(string context, IReadOnlyList<PyNTTDimExpression> actual, IReadOnlyList<PyNTTDimExpression> expected)
    {
        if (actual.Count > expected.Count)
        {
            throw new NotSupportedException($"{context} requires broadcastable shape, got [{ShapeText(actual)}] for output [{ShapeText(expected)}].");
        }

        var axisOffset = expected.Count - actual.Count;
        for (var i = 0; i < actual.Count; i++)
        {
            var actualDim = actual[i];
            var expectedDim = expected[axisOffset + i];
            if (!actualDim.IsFixedOne && !SameDim(actualDim, expectedDim))
            {
                throw new NotSupportedException($"{context} requires broadcastable shape, got [{ShapeText(actual)}] for output [{ShapeText(expected)}].");
            }
        }
    }

    private static void ValidateReduceShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> outputShape, IReadOnlyList<int> axes, bool keepDims)
    {
        var axisSet = axes.ToHashSet();
        if (keepDims)
        {
            if (outputShape.Count != inputShape.Count)
            {
                throw new NotSupportedException($"{context} with keep_dims expects output rank {inputShape.Count}, got [{ShapeText(outputShape)}].");
            }

            for (var axis = 0; axis < inputShape.Count; axis++)
            {
                var expected = axisSet.Contains(axis) ? PyNTTDimExpression.One : inputShape[axis];
                if (!SameDim(outputShape[axis], expected))
                {
                    throw new NotSupportedException($"{context} output shape mismatch at axis {axis}, expected {expected}, got {outputShape[axis]}.");
                }
            }

            return;
        }

        var expectedShape = inputShape
            .Select((dimension, axis) => (dimension, axis))
            .Where(item => !axisSet.Contains(item.axis))
            .Select(item => item.dimension)
            .ToArray();
        if (outputShape.Count != expectedShape.Length || outputShape.Zip(expectedShape).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} output shape mismatch, expected [{ShapeText(expectedShape)}], got [{ShapeText(outputShape)}].");
        }
    }

    private static void ValidatePadShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> outputShape, long[][] pads)
    {
        if (pads.Length != inputShape.Count || outputShape.Count != inputShape.Count)
        {
            throw new NotSupportedException($"{context} expects input, output, and pads to have the same rank.");
        }

        for (var axis = 0; axis < inputShape.Count; axis++)
        {
            var expected = AddDims(AddDims(inputShape[axis], ToDim(pads[axis][0])), ToDim(pads[axis][1]));
            if (!SameDim(outputShape[axis], expected))
            {
                throw new NotSupportedException($"{context} output shape mismatch at axis {axis}, expected {expected}, got {outputShape[axis]}.");
            }
        }
    }

    private static void ValidateSameRank(string context, IReadOnlyList<PyNTTDimExpression> lhsShape, IReadOnlyList<PyNTTDimExpression> rhsShape)
    {
        if (lhsShape.Count != rhsShape.Count)
        {
            throw new NotSupportedException($"{context} expects matching ranks, got {lhsShape.Count} and {rhsShape.Count}.");
        }
    }

    private static void ValidateSameRank(string context, int lhsRank, int rhsRank)
    {
        if (lhsRank != rhsRank)
        {
            throw new NotSupportedException($"{context} expects matching ranks, got {lhsRank} and {rhsRank}.");
        }
    }

    private static void ValidateScatterNDShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> indicesShape, IReadOnlyList<PyNTTDimExpression> updatesShape, IReadOnlyList<PyNTTDimExpression> outputShape)
    {
        if (inputShape.Count != outputShape.Count || inputShape.Zip(outputShape).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} expects input and output to have the same shape.");
        }

        if (indicesShape.Count < 1)
        {
            throw new NotSupportedException($"{context} expects indices rank >= 1.");
        }

        var indexDepth = RequireFixedDim(indicesShape[^1], $"{context} index depth");
        if (indexDepth < 1 || indexDepth > inputShape.Count)
        {
            throw new NotSupportedException($"{context} index depth {indexDepth} is out of range for input rank {inputShape.Count}.");
        }

        var expectedUpdatesShape = indicesShape.Take(indicesShape.Count - 1)
            .Concat(inputShape.Skip((int)indexDepth))
            .ToArray();
        if (updatesShape.Count != expectedUpdatesShape.Length || updatesShape.Zip(expectedUpdatesShape).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} updates shape mismatch, expected [{ShapeText(expectedUpdatesShape)}], got [{ShapeText(updatesShape)}].");
        }
    }

    private static (long[] Starts, long[] Strides) NormalizeSliceParameters(IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<long> starts, IReadOnlyList<int> axes, IReadOnlyList<int> strides, string context)
    {
        if (starts.Count != axes.Count || strides.Count != axes.Count)
        {
            throw new NotSupportedException($"{context} expects begins, axes, and strides to have the same length.");
        }

        var normalizedStarts = new long[inputShape.Count];
        var normalizedStrides = Enumerable.Repeat(1L, inputShape.Count).ToArray();
        var usedAxes = new HashSet<int>();
        for (var i = 0; i < axes.Count; i++)
        {
            var axis = NormalizeAxis(axes[i], inputShape.Count, context);
            if (!usedAxes.Add(axis))
            {
                throw new NotSupportedException($"{context} contains duplicated axis {axis}.");
            }

            var stride = strides[i];
            if (stride == 0)
            {
                throw new NotSupportedException($"{context} stride must not be zero.");
            }

            normalizedStarts[axis] = NormalizeSliceStart(starts[i], RequireFixedDim(inputShape[axis], $"{context} input dimension {axis}"), stride);
            normalizedStrides[axis] = stride;
        }

        return (normalizedStarts, normalizedStrides);
    }

    private static long NormalizeSliceStart(long start, long dim, long stride)
    {
        var normalized = start < 0 ? start + dim : start;
        if (stride > 0)
        {
            return Math.Clamp(normalized, 0, dim);
        }

        return Math.Clamp(normalized, 0, dim - 1);
    }

    private static long[] GetStaticRankedShape(BaseExpr expr, string context)
    {
        if (expr is not RankedShape shape)
        {
            throw new NotSupportedException($"{context} must be a static ranked shape.");
        }

        return shape.ToValueArray();
    }

    private static void ValidateGatherShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> indexShape, IReadOnlyList<PyNTTDimExpression> outputShape, int axis)
    {
        var expectedRank = inputShape.Count + indexShape.Count - 1;
        if (outputShape.Count != expectedRank)
        {
            throw new NotSupportedException($"{context} expects output rank {expectedRank}, got [{ShapeText(outputShape)}].");
        }

        for (var i = 0; i < axis; i++)
        {
            if (!SameDim(outputShape[i], inputShape[i]))
            {
                throw new NotSupportedException($"{context} output pre-axis shape mismatch, got [{ShapeText(outputShape)}] for input [{ShapeText(inputShape)}].");
            }
        }

        for (var i = 0; i < indexShape.Count; i++)
        {
            if (!SameDim(outputShape[axis + i], indexShape[i]))
            {
                throw new NotSupportedException($"{context} output index shape mismatch, got [{ShapeText(outputShape)}] for index [{ShapeText(indexShape)}].");
            }
        }

        for (var i = axis + 1; i < inputShape.Count; i++)
        {
            var outputAxis = i + indexShape.Count - 1;
            if (!SameDim(outputShape[outputAxis], inputShape[i]))
            {
                throw new NotSupportedException($"{context} output post-axis shape mismatch, got [{ShapeText(outputShape)}] for input [{ShapeText(inputShape)}].");
            }
        }
    }

    private static long[][] GetStaticPaddings(Paddings paddings, string context)
    {
        if (!paddings.IsFixed)
        {
            throw new NotSupportedException($"{context} requires fixed paddings.");
        }

        var pads = paddings.ToValueArray();
        return Enumerable.Range(0, paddings.Count)
            .Select(axis => new[] { pads[axis, 0], pads[axis, 1] })
            .ToArray();
    }

    private static void ValidateRank(string context, IReadOnlyList<PyNTTDimExpression> shape, int rank)
    {
        if (shape.Count != rank)
        {
            throw new NotSupportedException($"{context} requires rank {rank}, got [{ShapeText(shape)}].");
        }
    }

    private static void ValidateMinimumRank(string context, IReadOnlyList<PyNTTDimExpression> shape, int rank)
    {
        if (shape.Count < rank)
        {
            throw new NotSupportedException($"{context} requires rank >= {rank}, got [{ShapeText(shape)}].");
        }
    }

    private static void EnsureEmpty(string context, IRArray<int> values)
    {
        var valueArray = values.IsDefaultOrEmpty ? Array.Empty<int>() : values.ToArray();
        if (valueArray.Length > 0)
        {
            throw new NotSupportedException($"{context} must be empty for the current PyNTT Triton templates, got [{string.Join(",", valueArray)}].");
        }
    }

    private static string GetTritonDType(DataType dataType)
        => GetTritonDType(GetPyNTTDTypeName(GetScalarDataType(dataType)));

    private static string GetTritonDType(string dtypeName)
    {
        return dtypeName switch
        {
            "bool" => "tl.uint8",
            "int8" => "tl.int8",
            "uint8" => "tl.uint8",
            "int16" => "tl.int16",
            "uint16" => "tl.uint16",
            "int32" => "tl.int32",
            "uint32" => "tl.uint32",
            "int64" => "tl.int64",
            "uint64" => "tl.uint64",
            "float16" => "tl.float16",
            "bfloat16" => "tl.bfloat16",
            "float32" => "tl.float32",
            "float64" => "tl.float64",
            var name => throw new NotSupportedException($"Unsupported PyNTT Triton dtype: {name}."),
        };
    }

    private static string GetPyNTTScalarDTypeName(DataType dataType) => GetPyNTTDTypeName(GetScalarDataType(dataType));

    private static string GetScalarTritonDType(DataType dataType) => GetTritonDType(GetScalarDataType(dataType));

    private static DataType GetScalarDataType(DataType dataType)
    {
        return dataType switch
        {
            VectorType vectorType => vectorType.ElemType,
            MaskVectorType => DataTypes.Boolean,
            _ => dataType,
        };
    }

    private static int[] GetVectorLanes(DataType dataType)
    {
        return dataType switch
        {
            VectorType vectorType => vectorType.Lanes.ToArray(),
            MaskVectorType maskVectorType => new[] { maskVectorType.Lanes },
            _ => Array.Empty<int>(),
        };
    }

    private static int[][] GetSplitAxes(DistributedType distributedType)
    {
        var rank = distributedType.TensorType.Shape.Rank;
        return Enumerable.Range(0, rank)
            .Select(axis => axis < distributedType.AxisPolicies.Count && distributedType.AxisPolicies[axis] is SBPSplit split
                ? split.Axes.ToArray()
                : Array.Empty<int>())
            .ToArray();
    }

    private static bool IsObjectExpression(BaseExpr expr) => IsObjectDataType(expr.CheckedDataType);

    private static bool IsObjectDataType(DataType dataType) => dataType is ReferenceType;

    private static OutputInfo[] GetOutputInfos(BaseFunction function)
    {
        return BuildOutputInfos(PyNTTFunctionOutputs.GetOutputParameters(function));
    }

    private static OutputInfo[] BuildOutputInfos(IEnumerable<BufferVar> outputs)
    {
        return outputs
            .Select((output, index) =>
            {
                var type = output.CheckedType;
                var tensorType = GetTensorType(type, $"output{index}");
                return new OutputInfo(
                    $"output{index}",
                    output.Name,
                    GetRankedShape(tensorType, $"output{index}").Dimensions.ToArray()
                        .Select(dimension => new PyNTTDimExpressionEmitter().Emit(dimension))
                        .ToArray(),
                    tensorType.DType,
                    type as DistributedType);
            })
            .ToArray();
    }

    private static BaseExpr UnwrapInputBoxing(BaseExpr expr)
    {
        while (expr is Call call && call.Target is Boxing)
        {
            expr = call.Arguments[0];
        }

        return expr;
    }

    private static TensorType GetTensorType(IRType type, string name)
    {
        return type switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => throw new NotSupportedException($"PyNTT requires tensor type for {name}, got {type}."),
        };
    }

    private static RankedShape GetRankedShape(TensorType tensorType, string name)
    {
        if (tensorType.Shape is not RankedShape shape)
        {
            throw new NotSupportedException($"PyNTT requires ranked shape for {name}, got {tensorType.Shape}.");
        }

        return shape;
    }

    private static string GetPyNTTDTypeName(DataType dataType)
    {
        return dataType.GetDisplayName() switch
        {
            "bool" => "bool",
            "i8" => "int8",
            "u8" => "uint8",
            "i16" => "int16",
            "u16" => "uint16",
            "i32" => "int32",
            "u32" => "uint32",
            "i64" => "int64",
            "u64" => "uint64",
            "f16" => "float16",
            "bf16" => "bfloat16",
            "f32" => "float32",
            "f64" => "float64",
            var name => name,
        };
    }

    private static Dimension[] GetRankedShapeDimensions(Shape shape, string name)
    {
        if (shape is not RankedShape rankedShape)
        {
            throw new NotSupportedException($"PyNTT requires ranked shape for {name}, got {shape}.");
        }

        return rankedShape.Dimensions.ToArray();
    }

    private static PyNTTDimExpression Product(IEnumerable<PyNTTDimExpression> values)
    {
        var array = values.ToArray();
        if (array.Length == 0)
        {
            return PyNTTDimExpression.One;
        }

        long? fixedValue = array.All(value => value.FixedValue.HasValue)
            ? array.Aggregate(1L, (product, value) => checked(product * value.FixedValue!.Value))
            : null;
        return new(
            string.Join(" * ", array.Select(value => $"({value.PythonExpression})")),
            string.Join(" * ", array.Select(value => $"({value.TritonExpression})")),
            fixedValue);
    }

    private static string[] ToPythonExpressions(IEnumerable<PyNTTDimExpression> values)
        => values.Select(value => value.PythonExpression).ToArray();

    private static string[] GetRuntimeShapeArgs(GeneratedKernelMetadata kernel)
        => kernel.Attrs.TryGetValue("runtime_shape_args", out var value) && value is string[] args
            ? args
            : Array.Empty<string>();

    private static string SanitizePythonIdentifier(string value)
    {
        var chars = value.Select(ch => char.IsAsciiLetterOrDigit(ch) || ch == '_' ? ch : '_').ToArray();
        if (chars.Length == 0 || char.IsDigit(chars[0]))
        {
            return "_" + new string(chars);
        }

        return new string(chars);
    }

    private static string SanitizeBoundedPythonIdentifier(string value)
    {
        var sanitized = SanitizePythonIdentifier(value);
        if (sanitized.Length <= MaxGeneratedIdentifierLength)
        {
            return sanitized;
        }

        var digest = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sanitized)))
            .ToUpperInvariant()[..12];
        var separator = $"__h{digest}__";
        var retainedLength = MaxGeneratedIdentifierLength - separator.Length;
        var headLength = Math.Min(80, retainedLength / 2);
        var tailLength = retainedLength - headLength;
        return $"{sanitized[..headLength]}{separator}{sanitized[^tailLength..]}";
    }

    private sealed record OutputInfo(string Name, string AbiName, PyNTTDimExpression[] Shape, DataType DType, DistributedType? DistributedType);

    private sealed class SharedHelperRegistry
    {
        private readonly Dictionary<string, SharedHelperDefinition> _definitions = new(StringComparer.Ordinal);
        private int _nextDeviceFunctionIndex;
        private int _nextIndex;

        public string GetDeviceFunctionName(string ownerName, string calleeName)
        {
            var semanticCalleeName = calleeName.StartsWith("device_", StringComparison.Ordinal)
                ? calleeName
                : $"{calleeName}_device";
            return SanitizeBoundedPythonIdentifier($"{ownerName}_{semanticCalleeName}");
        }

        public string GetNextCodegenScopeDeviceFunctionName(string ownerName, string scopeName)
            => SanitizeBoundedPythonIdentifier(
                $"{ownerName}_device_{scopeName}_{_nextDeviceFunctionIndex++.ToString(CultureInfo.InvariantCulture)}");

        public bool TryGet(string key, out string name, out string[] arguments)
        {
            if (_definitions.TryGetValue(key, out var definition))
            {
                name = definition.Name;
                arguments = definition.Arguments;
                return true;
            }

            name = string.Empty;
            arguments = Array.Empty<string>();
            return false;
        }

        public string Add(string key, string ownerName, string kind)
        {
            var name = SanitizePythonIdentifier($"{ownerName}_shared_{kind}_{_nextIndex++}");
            _definitions.Add(key, new(name, Array.Empty<string>()));
            return name;
        }

        public void SetArguments(string key, string[] arguments)
        {
            if (!_definitions.TryGetValue(key, out var definition))
            {
                throw new NotSupportedException($"PyNTT shared helper registry does not contain key {key}.");
            }

            _definitions[key] = definition with { Arguments = arguments };
        }

        private sealed record SharedHelperDefinition(string Name, string[] Arguments);
    }
}
#pragma warning restore SA1201

internal sealed record PyNTTGeneratedKernelSource(
    [property: JsonPropertyName("generated_kernels")]
    IReadOnlyList<GeneratedKernelMetadata> Kernels,
    [property: JsonPropertyName("render_kernels")]
    IReadOnlyList<KernelRenderSpec> RenderKernels,
    [property: JsonPropertyName("source")]
    string Source);

internal sealed record GeneratedPrimFunctionKernel(
    GeneratedKernelMetadata Metadata,
    KernelRenderSpec RenderSpec);

internal sealed record KernelRenderSpec(
    [property: JsonPropertyName("metadata")]
    GeneratedKernelMetadata Metadata,
    [property: JsonPropertyName("helpers")]
    IReadOnlyList<HelperTemplateRenderSpec> Helpers,
    [property: JsonPropertyName("device_functions")]
    IReadOnlyList<DeviceFunctionRenderSpec> DeviceFunctions,
    [property: JsonPropertyName("body_source")]
    string BodySource);

internal sealed record KernelInputLayout(
    string[] Names,
    IReadOnlyDictionary<int, int> IndexMap,
    IReadOnlySet<int> RemovedIndexes,
    string BodySource,
    HelperTemplateRenderSpec[] Helpers,
    DeviceFunctionRenderSpec[] DeviceFunctions);

internal sealed record BufferViewSource(
    Nncase.TIR.Buffer Source,
    PyNTTDimExpression[] Offsets,
    PyNTTDimExpression[] Shape);

internal sealed record DeviceFunctionBuildResult(
    DeviceFunctionRenderSpec Function,
    IReadOnlyList<DeviceFunctionRenderSpec> NestedDeviceFunctions,
    IReadOnlyList<HelperKernelCallMetadata> HelperCalls,
    IReadOnlyList<string> OpKinds,
    bool RequiresGridBarrier,
    long BlockLocalDataPoolBytes,
    long SharedMemoryBytes,
    IReadOnlyDictionary<string, PyNTTKVCacheStorageMetadata?> FormalObjectFieldStorages,
    IReadOnlyDictionary<int, string> FormalObjectOutputAliases);

internal sealed record DeviceFunctionRenderSpec(
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("noinline")]
    bool NoInline,
    [property: JsonPropertyName("preserve_helper_call_boundaries")]
    bool PreserveHelperCallBoundaries,
    [property: JsonPropertyName("helpers")]
    IReadOnlyList<HelperTemplateRenderSpec> Helpers,
    [property: JsonPropertyName("body_source")]
    string BodySource,
    [property: JsonPropertyName("parameter_overrides")]
    IReadOnlyDictionary<string, string> ParameterOverrides,
    [property: JsonPropertyName("extra_parameters")]
    IReadOnlyList<string> ExtraParameters,
    [property: JsonPropertyName("extra_parameter_arguments")]
    IReadOnlyDictionary<string, string> ExtraParameterArguments);

internal sealed record HelperTemplateRenderSpec(
    [property: JsonPropertyName("template")]
    string Template,
    [property: JsonPropertyName("model")]
    object Model,
    [property: JsonPropertyName("arguments")]
    string[] Arguments,
    [property: JsonPropertyName("requires_inline")]
    bool RequiresInline);

internal sealed record GeneratedKernelMetadata(
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("op_kind")]
    string OpKind,
    [property: JsonPropertyName("inputs")]
    string[] Inputs,
    [property: JsonPropertyName("outputs")]
    string[] Outputs,
    [property: JsonPropertyName("attrs")]
    Dictionary<string, object> Attrs,
    [property: JsonPropertyName("launch")]
    LaunchMetadata Launch);

internal sealed record HelperKernelCallMetadata(
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("arguments")]
    string[] Arguments);

internal sealed record LaunchMetadata(
    [property: JsonPropertyName("meta")]
    Dictionary<string, object> Meta,
    [property: JsonPropertyName("tuning")]
    KernelTuningMetadata Tuning,
    [property: JsonPropertyName("sharding")]
    ShardMetadata Sharding,
    [property: JsonPropertyName("num_warps")]
    int? NumWarps,
    [property: JsonPropertyName("num_stages")]
    int? NumStages);

internal sealed record ShardMetadata(
    [property: JsonPropertyName("strategy")]
    string Strategy,
    [property: JsonPropertyName("placement_axis")]
    string PlacementAxis,
    [property: JsonPropertyName("tensor_axis")]
    int TensorAxis,
    [property: JsonPropertyName("extent")]
    string Extent,
    [property: JsonPropertyName("hierarchy")]
    int[] Hierarchy,
    [property: JsonPropertyName("hierarchy_levels")]
    string HierarchyLevels,
    [property: JsonPropertyName("global_shape")]
    string[] GlobalShape);

internal sealed record KernelTuningMetadata(
    [property: JsonPropertyName("parameters")]
    Dictionary<string, TuningParameterMetadata> Parameters);

internal sealed record TuningParameterMetadata(
    [property: JsonPropertyName("source")]
    string Source,
    [property: JsonPropertyName("candidates")]
    long[] Candidates);
