// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

public record BufferRenderInfo(string Name, string ElemType, int ElemSize, int Rank, ulong Offset, Dimension Size, bool IsFixedDimensions, bool IsFixedStrides, Dimension[] Dimensions, string DimensionsStr, Dimension[] Strides, string StridesStr, string? Distributed)
{
}

public sealed record KernelOutputRenderInfo(BufferVar Parameter, int Index, ulong Offset, ulong CapacityBytes);

public sealed record KernelDynamicDimensionBinding(DimVar Variable, int InputIndex, int DimensionIndex);

public sealed record KernelEntryAbiLayout(
    PrimFunctionAbiView Abi,
    IReadOnlyList<KernelOutputRenderInfo> Outputs,
    IReadOnlyList<KernelDynamicDimensionBinding> DynamicDimensions,
    ulong OutputAlignment,
    ulong OutputPoolSize)
{
    public static KernelEntryAbiLayout Create(PrimFunction function)
    {
        var abi = function.GetAbiView();
        if (abi.Workspaces.Count != 0)
        {
            throw new InvalidOperationException(
                $"NTT entry PrimFunction {function.Name} must receive workspaces through the runtime ABI, not formal parameters.");
        }

        var outputAlignment = Math.Max(
            function.SchedResult.OutputAlign,
            abi.OutputParameters
                .Select(parameter => checked((ulong)parameter.CheckedDataType.SizeInBytes))
                .DefaultIfEmpty(8UL)
                .Max());
        var outputs = new KernelOutputRenderInfo[abi.OutputParameters.Count];
        ulong outputPoolSize = 0;
        for (var index = 0; index < outputs.Length; index++)
        {
            var parameter = abi.OutputParameters[index];
            var capacityBytes = GetOutputCapacityBytes(parameter);
            var offset = MathUtility.AlignUp(outputPoolSize, outputAlignment);
            outputs[index] = new(parameter, index, offset, capacityBytes);
            outputPoolSize = checked(offset + capacityBytes);
        }

        outputPoolSize = MathUtility.AlignUp(outputPoolSize, outputAlignment);
        var dynamicDimensions = GetDynamicDimensionBindings(function, abi);
        ValidateOutputDimensions(function, abi, dynamicDimensions);
        return new(abi, outputs, dynamicDimensions, outputAlignment, outputPoolSize);
    }

    private static ulong GetOutputCapacityBytes(BufferVar parameter)
    {
        var tensorType = parameter.CheckedType switch
        {
            DistributedType distributedType => distributedType.TensorType,
            TensorType value => value,
            _ => throw new NotSupportedException(
                $"NTT output parameter {parameter.Name} must be tensor-like, got {parameter.CheckedType}."),
        };
        if (tensorType.Shape is not RankedShape shape || tensorType.DType is PointerType or ReferenceType)
        {
            throw new NotSupportedException(
                $"NTT output parameter {parameter.Name} must be a ranked value tensor, got {tensorType}.");
        }

        var strides = parameter.LayoutAnnotation.Kind switch
        {
            BufferLayoutKind.ExactStrided => parameter.LayoutAnnotation.Strides.ToArray(),
            BufferLayoutKind.RuntimeStrided => TensorUtilities.GetDefaultStrides(shape.Dimensions),
            _ => throw new NotSupportedException(
                $"NTT output parameter {parameter.Name} has unsupported layout {parameter.LayoutAnnotation.Kind}."),
        };
        var byteSpan = BufferViewUtility.GetByteSpanSize(shape.Dimensions, strides, tensorType.DType.SizeInBytes);
        var maxByteSpan = CompilerServices.GetMaxShape(new RankedShape([byteSpan]))[0];
        return checked((ulong)maxByteSpan);
    }

    private static IReadOnlyList<KernelDynamicDimensionBinding> GetDynamicDimensionBindings(
        PrimFunction function,
        PrimFunctionAbiView abi)
    {
        var explicitDimensions = abi.Inputs.OfType<DimVar>().Select(dimension => dimension.Name).ToHashSet(StringComparer.Ordinal);
        var bindings = new Dictionary<string, KernelDynamicDimensionBinding>(StringComparer.Ordinal);
        for (var inputIndex = 0; inputIndex < abi.Inputs.Count; inputIndex++)
        {
            var input = abi.Inputs[inputIndex];
            if (input.CheckedShape is not RankedShape shape)
            {
                continue;
            }

            for (var dimensionIndex = 0; dimensionIndex < shape.Rank; dimensionIndex++)
            {
                if (shape[dimensionIndex] is not DimVar dimension || explicitDimensions.Contains(dimension.Name))
                {
                    continue;
                }

                if (bindings.TryGetValue(dimension.Name, out var existing) &&
                    (!ReferenceEquals(existing.Variable, dimension) ||
                     existing.InputIndex != inputIndex ||
                     existing.DimensionIndex != dimensionIndex))
                {
                    throw new InvalidOperationException(
                        $"NTT entry PrimFunction {function.Name} has ambiguous runtime dimension {dimension.Name}.");
                }

                bindings.TryAdd(dimension.Name, new(dimension, inputIndex, dimensionIndex));
            }
        }

        return bindings.Values.OrderBy(binding => binding.Variable.Name, StringComparer.Ordinal).ToArray();
    }

    private static void ValidateOutputDimensions(
        PrimFunction function,
        PrimFunctionAbiView abi,
        IReadOnlyList<KernelDynamicDimensionBinding> dynamicDimensions)
    {
        var boundNames = abi.Inputs.OfType<DimVar>().Select(dimension => dimension.Name)
            .Concat(dynamicDimensions.Select(binding => binding.Variable.Name))
            .ToHashSet(StringComparer.Ordinal);
        foreach (var output in abi.OutputParameters)
        {
            var expressions = output.CheckedShape.ToArray().Cast<BaseExpr>()
                .Concat(output.LayoutAnnotation.Strides.Cast<BaseExpr>());
            foreach (var dimension in expressions.SelectMany(ExprCollector.Collect).OfType<DimVar>())
            {
                if (!boundNames.Contains(dimension.Name))
                {
                    throw new InvalidOperationException(
                        $"NTT entry PrimFunction {function.Name} cannot derive output dimension {dimension.Name} " +
                        $"for {output.Name} from its runtime inputs.");
                }
            }
        }
    }
}

public record KernelMainModel(TIR.PrimFunction PrimFunction, NTTTargetOptions Options, ulong Alignment, ulong DataSize, ulong BlockLocalDataPoolSize, ulong RDataSize, ulong BlockLocalRdataPoolSize)
{
    public KernelEntryAbiLayout EntryAbi { get; } = KernelEntryAbiLayout.Create(PrimFunction);

    public BufferRenderInfo GetInfo(TIR.Buffer buffer)
    {
        ulong offset = ((TensorConst)buffer.MemSpan.Buffer.Start).Value.ToScalar<ulong>() + (ulong)buffer.MemSpan.Start.FixedValue;

        var elemType = buffer.ElemType.ToC();
        var rank = buffer.Dimensions.Length;
        var size = buffer.MemSpan.Size / buffer.ElemType.SizeInBytes;
        var isFixedDims = buffer.Dimensions.AsValueEnumerable().All(d => d.IsFixed);
        var isFixedStrides = buffer.Strides.AsValueEnumerable().All(d => d.IsFixed);
        var dims = KernelUtility.DimensionsTypeToC(isFixedDims, buffer.Dimensions);
        var strides = KernelUtility.StridesTypeToC(isFixedStrides, buffer.Strides);
        var distributed = buffer.DistributedType == null ? null : KernelUtility.ShardingToC(buffer.DistributedType);
        return new(buffer.Name, elemType, buffer.ElemType.SizeInBytes, rank, offset, size, isFixedDims, isFixedStrides, buffer.Dimensions.ToArray(), dims, buffer.Strides.ToArray(), strides, distributed);
    }

    public string RenderDimension(Dimension dimension) => new CSourceConvertVisitor().Visit(dimension).Name;
}

public record NTTTargetOptionsModel(NTTTargetOptions Options, ulong Alignment, ulong CollectivePoolSize)
{
}

public static class CSourceBuiltn
{
    public const string DeviceHeader = @"#pragma once
#include <nncase/ntt/ntt.h>
#include ""topo_aware_runtime.h""
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public const string KernelDeclareHeader = @"#pragma once
#include <nncase/ntt/ntt.h>
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public const string KernelHeader = @"#pragma once
#include <nncase/ntt/ntt.h>
#include ""lambda_functions.h""
#include ""device_functions.h""
#include ""kernel_functions.h""
#include ""topo_aware_runtime.h""
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public const string ThreadMainHeader = @"#include <nncase/ntt/ntt.h>
#include ""kernel_functions.h""
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;

";

    public static string TopoAwareRuntimeDef(NTTTargetOptions options, ulong dataAlign, ulong collective_pool_size)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/topo_aware_runtime.cshtml", new NTTTargetOptionsModel(options, dataAlign, collective_pool_size)).Result;
        return content;
    }

    public static string ModuleTopologyDef(NTTTargetOptions options, bool isCUDA)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/module_topology_def.h.cshtml", new { Options = options, IsCUDA = isCUDA }).Result;
        return content;
    }

    public static string CMakeDef(bool isCUDA)
    {
        var cmakePath = CMakePath(Path.Combine(Path.GetDirectoryName(typeof(CSourceBuiltn).Assembly.Location)!, "Runtime", "cmake", "ntt_module.cmake"));
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/CMakeLists.txt.cshtml", new { CMakePath = cmakePath, IsCUDA = isCUDA }).Result;
        return content;
    }

    public static string MakeMain(TIR.PrimFunction primFunction, ulong dataAlign, ulong dataUsage, ulong blockLocalDataPoolSize, ulong rdataPoolSize, ulong blockLocalRdataPoolSize, NTTTargetOptions options)
    {
        var content = RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/thread_main.cpp.cshtml", new KernelMainModel(primFunction, options, dataAlign, dataUsage, blockLocalDataPoolSize, rdataPoolSize, blockLocalRdataPoolSize)).Result;
        return content;
    }

    public static string MakeKernel(string ctype, string kernelImpl)
    {
        return KernelHeader + ctype + kernelImpl;
    }

    private static string CMakePath(string path) =>
        path.Replace("\\", "/", StringComparison.Ordinal);
}
