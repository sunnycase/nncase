// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

// #define PROFILE_CALL
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

/// <summary>
/// convert single prim function to c source.
/// </summary>
internal sealed class KernelCSourceConvertVisitor : CSourceConvertVisitor, IDisposable
{
    private readonly StringBuilder _kernelBuilder;
    private readonly HashSet<TIR.PrimFunction> _refFuncs;
    private readonly HashSet<TIR.Buffer> _declaredBuffers = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<string> _localNames = new(StringComparer.Ordinal)
    {
        "rdata",
        "block_local_rdata",
        "data",
        "block_local_data",
        "output",
        "output_descs",
    };

    private readonly Dictionary<string, int> _localNameSuffixes = new(StringComparer.Ordinal);
    private readonly ulong _chipLocalRdataBase;
    private IVar[]? _tensorParams;

    public KernelCSourceConvertVisitor(NTTTargetOptions targetOptions, ulong chipLocalRdataBase = 0)
    {
        _kernelBuilder = new StringBuilder();
        _refFuncs = new(ReferenceEqualityComparer.Instance);
        _chipLocalRdataBase = chipLocalRdataBase;
        CollectivePoolSize = 0;
        TargetOptions = targetOptions;
    }

    public NTTTargetOptions TargetOptions { get; }

    public ulong CollectivePoolSize { get; private set; }

    private IVar[] TensorParams => _tensorParams ??= VisitEntry.Parameters.ToArray().Where(IsTemplateTensorParameter).ToArray();

    public void WriteWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal); // 找到第一个 '(' 的位置
            if (index != -1)
            {
                tagName = functionName.Substring(0, index); // 截取从头到 '(' 之前的部分
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
        IndentScope.Writer.Write($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.Write($"profile_scope profiler(0, profile_level::kernel);\n");
        IndentScope.Writer.Write($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

    public void WriteIndWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal); // 找到第一个 '(' 的位置
            if (index != -1)
            {
                tagName = functionName.Substring(0, index); // 截取从头到 '(' 之前的部分
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
        IndentScope.Writer.IndWrite($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.IndWrite($"profile_scope profiler(0, profile_level::kernel);\n");
        IndentScope.Writer.IndWrite($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

    public KernelCSource GetCSource()
    {
        var runtimeParameters = VisitEntry.Parameters.ToArray().Where(IsRuntimeParameter).ToArray();
        var templateHeader = TensorParams.Length == 0 ? string.Empty : $"template<{string.Join(", ", Enumerable.Range(0, TensorParams.Length).Select(x => $"class T{x}"))}>" + Environment.NewLine;
        var ctype = templateHeader +
            $"NTT_DEVICE void {VisitEntry.Name}({string.Concat(runtimeParameters.Select(Visit).Select(s => $"{s.Type} {s.Name}, ").ToArray())}const std::byte *rdata, const std::byte *block_local_rdata, std::byte *data, std::byte *block_local_data, std::byte *output, nncase::ntt::runtime::block_inout_desc *const output_descs)";
        return new(
            Declare: ctype + ";\n",
            Kernel: CSourceBuiltn.MakeKernel(ctype, _kernelBuilder.ToString()),
            CollectivePoolSize: CollectivePoolSize);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
    }

    private static bool IsRuntimeParameter(IVar parameter)
        => parameter is not BufferVar { Role: BufferVarRole.Workspace };

    private static bool IsTemplateTensorParameter(IVar parameter)
        => IsRuntimeParameter(parameter) && parameter.CheckedType is TensorType or DistributedType;

    protected override CSymbol VisitVar(Var expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var name = AllocateLocalName(IRHelpers.GetIdentityName(expr.Name));
        var index = Array.IndexOf(TensorParams, expr);
        if (index != -1)
        {
            symbol = new CSymbol($"T{index}", name);
        }
        else
        {
            symbol = new(expr.CheckedDataType.ToC(), name);
        }

        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitBufferVar(BufferVar expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var name = AllocateLocalName(IRHelpers.GetIdentityName(expr.Name));
        var index = Array.IndexOf(TensorParams, expr);
        if (index != -1)
        {
            symbol = new CSymbol($"T{index}", name);
        }
        else
        {
            symbol = new(expr.CheckedDataType.ToC(), name);
        }

        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitPrimFunction(PrimFunction expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var returnType = ((CallableType)expr.CheckedType).ReturnType;
        var ctype = $"void {expr.Name}({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit).Select(s => $"{s.Type} {s.Name}").ToArray())})";

        using (var scope = new IndentScope(_kernelBuilder))
        {
            // 1. Function signature
            IndentScope.Writer.IndWrite($"{{\n");

            WriteDimVars();

            // 3. Function body
            using (_ = new IndentScope())
            {
                Visit(expr.Body);
                Visit(expr.Results);
            }

            // 4. Function closing
            IndentScope.Writer.IndWrite("}\n");
        }

        symbol = new(ctype, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitFusion(Fusion fusion)
    {
        if (_exprMemo.TryGetValue(fusion, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = fusion.Name;
        symbol = new(type, str);
        _exprMemo.Add(fusion, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitPhysicalBuffer(PhysicalBuffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        if (expr.Start is Call { Target: IR.Buffers.AddressOf })
        {
            var buffers = expr.Users.OfType<TIR.MemSpan>()
                .SelectMany(memSpan => memSpan.Users.OfType<TIR.Buffer>())
                .Select(buffer => buffer.Name)
                .Distinct(StringComparer.Ordinal)
                .ToArray();
            throw new InvalidOperationException(
                $"NTT C codegen requires bufferized physical storage, but {expr.Location} buffer(s) [{string.Join(", ", buffers)}] still use AddressOf.");
        }

        var start = Visit(expr.Start);
        var isReadOnly = expr.Start.CheckedDataType is not ReferenceType &&
            (expr.Location is MemoryLocation.Rdata or MemoryLocation.ChipLocalRdata or MemoryLocation.BlockLocalRdata ||
             (expr.Location == MemoryLocation.Input && expr.Start is not BufferVar { Role: BufferVarRole.InOut }));
        if (expr.Location is MemoryLocation.Input or MemoryLocation.Output)
        {
            if (expr.Start is not BufferVar)
            {
                throw new InvalidOperationException(
                    $"NTT C codegen requires {expr.Location} storage to be backed by a BufferVar, but found {expr.Start.GetType().Name}.");
            }

            var byteType = isReadOnly ? "const std::byte" : "std::byte";
            var spanSize = Visit(expr.Size).Name;
            var parameterSpan = $"span_cast<{byteType}>({start.Name}.elements())";
            var parameterName = $"make_subspan({parameterSpan}, 0_dim, {spanSize})";
            symbol = new(start.Type, parameterName);
            _exprMemo.Add(expr, symbol);
            return symbol;
        }

        var ptypeName = isReadOnly ? "const std::byte" : "std::byte";
        if (expr.Location == MemoryLocation.ChipLocalData)
        {
            if (!expr.Size.IsFixed || expr.Size.FixedValue != 0)
            {
                throw new NotSupportedException(
                    "NTT CPU runtime does not provide a chip-local mutable workspace.");
            }

            symbol = new(start.Type, $"ntt::span<{ptypeName}, 0>()");
            _exprMemo.Add(expr, symbol);
            return symbol;
        }

        string loc = (expr.Location, expr.Hierarchy) switch
        {
            (MemoryLocation.Rdata, 0) => "rdata",
            (MemoryLocation.ChipLocalRdata, 0) => "rdata",
            (MemoryLocation.BlockLocalRdata, 0) => "block_local_rdata",
            (MemoryLocation.Data, 0) => "data",
            (MemoryLocation.Data, 1) => "data",
            (MemoryLocation.BlockLocalData, 0) => "block_local_data",
            _ => throw new NotSupportedException($"{expr.Location}, {expr.Hierarchy}"),
        };

        string name;
        var startName = expr.Location == MemoryLocation.ChipLocalRdata
            ? $"{_chipLocalRdataBase}UL + {start.Name}UL"
            : $"{start.Name}UL";
        if (expr.Size is DimConst)
        {
            var spanSize = (ulong)expr.Size.FixedValue;
            name = $"ntt::span<{ptypeName}, {spanSize}>({loc} + {startName}, {spanSize})";
        }
        else
        {
            var spanSize = Visit(expr.Size).Name;
            name = $"ntt::span<{ptypeName}>({loc} + {startName}, {spanSize})";
        }

        symbol = new(start.Type, name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitMemSpan(MemSpan expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var buffer = Visit(expr.Buffer);
        var start = Visit(expr.Start);

        var spanSize = Visit(expr.Size).Name;
        var name = $"make_subspan({buffer.Name}, {start.Name}, {spanSize})";

        symbol = new(start.Type, name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitBuffer(TIR.Buffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var dimensions = expr.DistributedType is null ? expr.Dimensions : ((RankedShape)expr.DistributedType.TensorType.Shape).Dimensions;
        var dimensionTypes = dimensions.AsValueEnumerable().Select(x => Visit(x).Type).ToArray();
        var strideTypes = expr.Strides.AsValueEnumerable().Select(x => Visit(x).Type).ToArray();
        var dtypeStr = GetBufferElementType(expr);
        var dimensionStr = $"shape_t<{StringUtility.Join(", ", dimensionTypes)}>";
        var strideStr = $"strides_t<{StringUtility.Join(", ", strideTypes)}>";

        var type = expr.MemSpan.Buffer.Location is MemoryLocation.Rdata or MemoryLocation.ChipLocalRdata or MemoryLocation.BlockLocalRdata || expr.MemSpan.Buffer.Start is TensorConst
            ? (expr.DistributedType == null
             ? $"tensor_view<{dtypeStr}, {dimensionStr}, {strideStr}> "
             : $"sharded_tensor_view<{dtypeStr}, {dimensionStr}, {KernelUtility.ShardingToC(expr.DistributedType)}, {strideStr}> ")
            : $"tensor<{dtypeStr}, {dimensionStr}, {strideStr}> ";

        symbol = new(type, AllocateLocalName(expr.Name));
        DeclBuffer(expr, symbol);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitCall(Call expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        string type = expr.CheckedType switch
        {
            TupleType x when x == TupleType.Void => string.Empty,
            TensorType { IsScalar: true } x => x.DType.ToC(),
            _ => throw new NotSupportedException(),
        };

        string str = string.Empty;
        if (expr.Target is TIR.ChannelProduce produce)
        {
            var args = expr.Arguments.ToArray();
            if (args.Length != 2 || args[1] is not TIR.Buffer value)
            {
                throw new NotSupportedException(
                    $"NTT ChannelProduce {produce.ChannelId} expects a channel and one TIR buffer payload.");
            }

            Visit(value);
            WriteIndWithProfiler(
                $"ntt::runtime::pipeline_channel_produce({Visit(args[0]).Name}, {VisitBuffer(value, local: false).Name}, {(produce.Phase + 1).ToString(System.Globalization.CultureInfo.InvariantCulture)}U)",
                "pipeline_channel_produce");
        }
        else if (expr.Target is TIR.ChannelConsume consume)
        {
            var args = expr.Arguments.ToArray();
            if (args.Length != 2 || args[1] is not TIR.Buffer destination)
            {
                throw new NotSupportedException(
                    $"NTT ChannelConsume {consume.ChannelId} expects a channel and one TIR buffer destination.");
            }

            Visit(destination);
            WriteIndWithProfiler(
                $"ntt::runtime::pipeline_channel_consume({Visit(args[0]).Name}, {VisitBuffer(destination, local: false).Name}, {(consume.Phase + 1).ToString(System.Globalization.CultureInfo.InvariantCulture)}U)",
                "pipeline_channel_consume");
        }
        else if (expr.Target is Op kop && kop is TIR.NTT.NTTKernelOp or TIR.Memcopy)
        {
            foreach (var item in expr.Arguments.ToArray().OfType<TIR.Buffer>())
            {
                Visit(item);
            }
#if PROFILE_CALL
            IndentScope.Writer.Write($"auto start_{CallCount} = get_ms_time();\n");
#endif
            var args = expr.Arguments.ToArray();
            if (kop is TIR.NTT.NTTKernelOp)
            {
                if (args.Length == 0 || args[^1] is not None)
                {
                    throw new NotSupportedException(
                        $"NTT CPU kernel {kop.GetType().Name} requires a None shared_workspace operand.");
                }

                args = args[..^1];
            }

            if (args.Any(x => x is TIR.Buffer { MemSpan: { Buffer: { Location: MemoryLocation.BlockLocalData } } }))
            {
                // Ensure all threads reach this point before a kernel using BlockLocalData
                WriteIndWithProfiler($"ntt::distributed::topology_synchronize<ntt::distributed::topology::block>();\n");
            }

            switch (kop)
            {
                case TIR.NTT.Unary unary:
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        UnaryOp = unary.UnaryOp,
                    }).Result);
                    break;
                case TIR.NTT.TensorLoad load:
                    if (args.Length == 1)
                    {
                        var fullShape = args[0].CheckedShape.ToValueArray();
                        (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                        CollectivePoolSize = Math.Max(CollectivePoolSize, (ulong)maxSize);
                        var indices = args[0].CheckedShape.Select(e => Visit(e).Name).ToSlicing(load.NdSbp, load.Placement)[0];
                        WriteWithProfiler($"tac::tensor_boxing_load_sync<fixed_shape_t<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
                    }
                    else
                    {
                        WriteWithProfiler($"reshard({VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[0], local: false).Name});\n");
                    }

                    break;
                case TIR.NTT.TensorStore store:
                    if (args.Length == 1)
                    {
                        var fullShape = args[0].CheckedShape.ToValueArray();
                        (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                        CollectivePoolSize = Math.Max(CollectivePoolSize, (ulong)maxSize);
                        var indices = args[0].CheckedShape.Select(e => Visit(e).Name).ToSlicing(store.NdSbp, store.Placement)[0];
                        WriteWithProfiler($"tac::tensor_boxing_store_sync<fixed_shape_t<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
                    }
                    else
                    {
                        (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                        CollectivePoolSize = Math.Max(CollectivePoolSize, (ulong)maxSize);
                        WriteWithProfiler($"reshard({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    }

                    break;
                case TIR.NTT.Im2col im2col:
                    WriteIndWithProfiler($"im2col({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape_v<{string.Join(",", im2col.Kernel)}>, fixed_shape_v<{string.Join(",", im2col.Stride)}>, fixed_paddings_v<{string.Join(",", im2col.Padding)}>, fixed_shape_v<{string.Join(",", im2col.VectorizedAxes)}>, fixed_shape_v<{string.Join(",", im2col.PadedNums)}>);\n");
                    break;
                case TIR.NTT.Pack vectorize:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Pack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Pack>(vectorize)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;

                case TIR.NTT.Unpack devectorize:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unpack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Unpack>(devectorize)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;
                case TIR.NTT.VectorizedLayerNorm vectorizedLayerNorm:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/VectorizedLayerNorm.cshtml", new TypedKernelTemplateModel<TIR.NTT.VectorizedLayerNorm>(vectorizedLayerNorm)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).Concat(vectorizedLayerNorm.PadedNums.Select(Visit).Select(x => new KernelArgument { Symbol = x })).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.NormStats normStats:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/NormStats.cshtml", new TypedKernelTemplateModel<TIR.NTT.NormStats>(normStats)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: false) }).ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.NormApply normApply:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/NormApply.cshtml", new TypedKernelTemplateModel<TIR.NTT.NormApply>(normApply)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: false) }).ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.InstanceNorm instanceNorm:
                    WriteWithProfiler($"instance_norm({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {args[0].CheckedDataType.ToC()} {{ {instanceNorm.Epsilon} }}, fixed_shape_v<{string.Join(",", instanceNorm.VectorizedAxes)}>{{}}, fixed_shape_v<{string.Join(",", instanceNorm.PadedNums)}>{{}} );\n");
                    break;
                case TIR.NTT.ResizeImage resize:
                    WriteIndWithProfiler($"resize({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape_v<{string.Join(",", resize.VectorizedAxes)}>, fixed_shape_v<{string.Join(",", resize.PadedNums)}>, fixed_shape_v<{string.Join(",", resize.NewSize)}>, image_resize_mode_t::{resize.ResizeMode.ToC()}, image_resize_transformation_mode_t::{resize.TransformationMode.ToC()}, image_resize_nearest_mode_t::{resize.NearestMode.ToC()});\n");
                    break;
                case TIR.NTT.VectorizedSoftmax vectorizedsoftmax:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/VectorizedSoftmax.cshtml", new TypedKernelTemplateModel<TIR.NTT.VectorizedSoftmax>(vectorizedsoftmax)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.VectorizedBinary vectorizedBinary:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                        {
                            BinaryOp = vectorizedBinary.BinaryOp,
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.Conv2D conv:
                    WriteIndWithProfiler($"conv2d({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, fixed_shape_v<{string.Join(", ", conv.Stride)}>, fixed_paddings_v<{string.Join(", ", conv.Padding)}>, fixed_shape_v<{string.Join(",", conv.Dilation)}>, {conv.Groups}_dim);\n");
                    break;
                case TIR.NTT.Matmul matmul:
                    {
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", new TypedKernelTemplateModel<TIR.NTT.Matmul>(matmul)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            }).Result,
                            "matmul");
                    }

                    break;
                case TIR.NTT.SUMMA summa:
                    {
                        var scale = Visit(args[3]).Name;
                        var rdKind = "tar::reduce_kind::" + string.Join("_", Enumerable.Range(0, TargetOptions.HierarchyNames.Length).Select(i => i >= TargetOptions.HierarchyNames.Length - 2 ? "r" + TargetOptions.HierarchyNames[i] : string.Empty + TargetOptions.HierarchyNames[i]));
                        IndentScope.Writer.IndWrite($"{{tac::detail::tensor_reduce_sync_impl<reduce_op::sum, {rdKind}> impl; impl.reduce_group_sync();\n");
                        IndentScope.Writer.IndWrite($"summa<false>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name}, {VisitBuffer(args[2], local: false).Name}, {scale}, fixed_shape_v<{string.Join(",", summa.LhsVectorizedAxes)}>, fixed_shape_v<>, fixed_shape_v<{string.Join(",", summa.RhsVectorizedAxes)}>, fixed_shape_v<>);\n");
                        IndentScope.Writer.IndWrite($"impl.reduce_group_sync();}}\n");
                    }

                    break;
                case TIR.NTT.PackedMatMul matmul:
                    {
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedMatMul.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedMatMul>(matmul)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            }).Result,
                            "packed_matmul");
                    }

                    break;
                case TIR.NTT.PackedMatMulNormStats matmulNormStats:
                    {
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedMatMulNormStats.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedMatMulNormStats>(matmulNormStats)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            }).Result,
                            "packed_matmul_norm_stats");
                    }

                    break;
                case TIR.NTT.BlockScaledMatMul blockScaledMatMul:
                    {
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/BlockScaledMatMul.cshtml", new TypedKernelTemplateModel<TIR.NTT.BlockScaledMatMul>(blockScaledMatMul)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: false) }).ToArray(),
                            }).Result,
                            "block_scaled_matmul");
                    }

                    break;
                case TIR.NTT.QKVParallelLinear qkvParallelLinear:
                    {
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/QKVParallelLinear.cshtml", new TypedKernelTemplateModel<TIR.NTT.QKVParallelLinear>(qkvParallelLinear)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            }).Result,
                            "qkv_parallel_linear");
                    }

                    break;
                case TIR.NTT.PackedQKVParallelLinear packedQKVParallelLinear:
                    {
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedQKVParallelLinear.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedQKVParallelLinear>(packedQKVParallelLinear)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            }).Result,
                            "packed_qkv_parallel_linear");
                    }

                    break;
                case TIR.NTT.MatMulGlu matMulGlu:
                    {
                        ValidateMatMulGluScales(args);
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/MatMulGlu.cshtml", new TypedKernelTemplateModel<TIR.NTT.MatMulGlu>(matMulGlu)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            }).Result,
                            "matmul_glu");
                    }

                    break;
                case TIR.NTT.PackedMatMulGlu packedMatMulGlu:
                    {
                        ValidateMatMulGluScales(args);
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedMatMulGlu.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedMatMulGlu>(packedMatMulGlu)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            }).Result,
                            "packed_matmul_glu");
                    }

                    break;
                case TIR.Memcopy copy:
                    WriteWithProfiler($"tensor_copy_sync({VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[0], local: true).Name});\n");
                    break;
                case TIR.NTT.Gather gather:
                    {
                        WriteWithProfiler($"gather({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {gather.Axis}_dim);\n");
                    }

                    break;
                case TIR.NTT.Swish swish:
                    if (swish.Beta != 1.0f)
                    {
                        IndentScope.Writer.IndWrite($"\n{{\nauto b= {swish.Beta}; auto tb = make_tensor_view_from_address<float>(&b, fixed_shape_v<>);\n");
                        WriteIndWithProfiler($"binary<ops::swishb>({VisitBuffer(args[0], local: true).Name}, tb, {VisitBuffer(args[1], local: true).Name});\n}}\n");
                    }
                    else
                    {
                        WriteWithProfiler($"unary<ops::swish>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    }

                    break;
                case TIR.NTT.Slice slice:
                    WriteWithProfiler($"slice({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {VisitDimOrShape(args[1]).Name}, {VisitDimOrShape(args[2]).Name}, fixed_dims_v<{string.Join(",", slice.Axes)}>, fixed_dims_v<{string.Join(",", slice.Strides)}>);\n");
                    break;
                case TIR.NTT.Concat concat:
                    WriteWithProfiler($"concat(ntt::make_tuple({string.Join(",", args.SkipLast(1).Select(x => VisitBuffer(x, local: true)).Select(s => s.Name))}), {VisitBuffer(args[^1], local: true).Name}, {concat.Axis}_dim);\n");
                    break;
                case TIR.NTT.Transpose transpose:
                    WriteWithProfiler($"transpose({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_dims_v<{string.Join(",", transpose.Perm)}>);\n");
                    break;
                case TIR.NTT.Pad pad:
                    var padValueType = args[0].CheckedTensorType.DType is VectorType vt ? vt.ElemType : args[0].CheckedTensorType.DType;
                    WriteWithProfiler($"pad({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitDimOrShape(args[1]).Name}, {args[0].CheckedDataType.ToC()} {{ ({padValueType.ToC()}){pad.PadValue} }});\n");
                    break;
                case TIR.NTT.Reduce reduce:
                    WriteWithProfiler($"reduce_{reduce.ReduceOp.ToC()}({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape_v<{string.Join(",", reduce.Axes)}>, fixed_shape_v<{string.Join(",", reduce.VectorizedAxes)}>);\n");
                    break;
                case TIR.NTT.ReduceArg reduceArg:
                    WriteWithProfiler($"reduce_arg<ops::{reduceArg.ReduceArgOp.ToC()[4..]}, {reduceArg.Axis}, {reduceArg.SelectLastIndex.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}, {reduceArg.KeepDims.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.Clamp clamp:
                    string min = clamp.Min is float.NegativeInfinity ? float.MinValue.ToString() : clamp.Min.ToString();
                    string max = clamp.Max is float.PositiveInfinity ? float.MaxValue.ToString() : clamp.Max.ToString();
                    WriteWithProfiler($"clamp({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, (float){min}, (float){max});\n");
                    break;
                case TIR.NTT.Cast cast:
                    {
                        string postOps = string.Empty;
                        if (expr[TIR.NTT.Cast.PostOps] is Fusion lambda)
                        {
                            postOps = $"<{lambda.Name}>";
                        }

                        WriteWithProfiler($"cast{postOps}({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape_v<{string.Join(",", cast.VectorizeAxes.ToArray())}>);\n");
                    }

                    break;
                case TIR.NTT.Where where:
                    WriteWithProfiler($"where({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");
                    break;
                case TIR.NTT.Expand expand:
                    WriteWithProfiler($"expand({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.Erf erf:
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        UnaryOp = UnaryOp.Erf,
                    }).Result);
                    break;
                case TIR.NTT.Compare compare:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Compare.cshtml", new CompareKernelTemplateModel
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            CompareOp = compare.CompareOp,
                        }).Result);
                    }

                    break;
                case TIR.NTT.ScatterND scatterND:
                    WriteWithProfiler($"scatter_nd({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");

                    break;
                case TIR.NTT.GatherReduceScatter grs:
                    {
                        if (grs.InType.Partial is not null)
                        {
                            var sbpPartial = grs.InType.Partial;
                            var reduceKind = "tar::reduce_kind::" + string.Join("_", Enumerable.Range(0, TargetOptions.HierarchyNames.Length).Select(i => (sbpPartial.Axes.Contains(i) ? "r" : string.Empty) + TargetOptions.HierarchyNames[i]));
                            WriteIndWithProfiler($"tac::tensor_reduce_sync<reduce_op::{sbpPartial.Op.ToC()}, {reduceKind}>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name});\n");
                        }
                        else
                        {
                            (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                            CollectivePoolSize = Math.Max(CollectivePoolSize, (ulong)maxSize);
                            WriteWithProfiler($"reshard({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name});\n");
                        }
                    }

                    break;
                case TIR.NTT.GetItem getItem:
                    IndentScope.Writer.Write($"get_item({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitDimOrShape(args[1]).Name});\n");
                    break;
                case TIR.NTT.GetPositionIds getPositionIds:
                    WriteIndWithProfiler($"get_position_ids({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {KernelUtility.ShardingToC(getPositionIds.DistributedType)}, {Visit(getPositionIds.DistributedType.TensorType.Shape).Name});\n");
                    break;
                case TIR.NTT.Stack stack:
                    IndentScope.Writer.Write($"stack<{stack.Axis}>(ntt::make_tuple({string.Join(",", args.SkipLast(1).Select(x => VisitBuffer(x, local: true)).Select(s => s.Name))}), {VisitBuffer(args[^1], local: true).Name});\n");
                    break;
                case TIR.NTT.ShapeOf shapeOf:
                    IndentScope.Writer.Write($"shapeof({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.Range range:
                    IndentScope.Writer.Write($"range({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");
                    break;
                case TIR.NTT.RoPE rope:
                    IndentScope.Writer.Write($"rope({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");
                    break;
                case TIR.NTT.ConstantOfShape constantOfShape:
                    IndentScope.Writer.Write($"constant_of_shape({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.NTT.UpdatePagedAttentionKVCache updatePagedAttentionKVCache:
                    WriteIndWithProfiler($"update_paged_attention_kv_cache<caching::attention_cache_kind::{updatePagedAttentionKVCache.CacheKind.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {Visit(args[2]).Name}, {updatePagedAttentionKVCache.Layout.ToC()});\n");
                    break;
                case TIR.NTT.QKVRoPEWithCache qkvRoPEWithCache:
                    WriteIndWithProfiler($"qkv_rope_with_cache<{qkvRoPEWithCache.QUseMean.ToString().ToLowerInvariant()}, {qkvRoPEWithCache.KUseMean.ToString().ToLowerInvariant()}>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name}, {VisitBuffer(args[2], local: false).Name}, {VisitBuffer(args[3], local: false).Name}, {VisitBuffer(args[4], local: false).Name}, {VisitBuffer(args[5], local: false).Name}, {VisitBuffer(args[6], local: false).Name}, {VisitBuffer(args[7], local: false).Name}, {VisitBuffer(args[8], local: false).Name}, {VisitBuffer(args[9], local: true).Name}, {Visit(args[10]).Name}, {VisitBuffer(args[11], local: false).Name}, {qkvRoPEWithCache.QEpsilon}f, {qkvRoPEWithCache.KEpsilon}f, {qkvRoPEWithCache.QKVLayout.ToC()}, {qkvRoPEWithCache.AttentionLayout.ToC()});\n");
                    break;
                case TIR.NTT.GatherPagedAttentionKVCache gakv:
                    IndentScope.Writer.IndWrite($"gather_paged_attention_kv_cache({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.NTT.PagedAttention pagedAttention:
                    var outputGate = args[5] is None
                        ? "nullptr"
                        : VisitBuffer(args[5], local: true).Name;
                    WriteIndWithProfiler($"paged_attention({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {Visit(args[4]).Name}, {outputGate}, {VisitBuffer(args[6], local: true).Name}, {pagedAttention.Layout.ToC()});\n");
                    break;
                case TIR.NTT.SynchronizeThreads:
                    WriteIndWithProfiler($"ntt::distributed::topology_synchronize<ntt::distributed::topology::block>();\n");
                    break;
                case TIR.NTT.Barrier barrier:
                    var topology = barrier.Scope switch
                    {
                        TIR.NTT.BarrierScope.Block => "block",
                        TIR.NTT.BarrierScope.Chip => "chip",
                        _ => throw new NotSupportedException($"Unsupported NTT barrier scope: {barrier.Scope}."),
                    };
                    WriteIndWithProfiler($"ntt::distributed::topology_synchronize<ntt::distributed::topology::{topology}>();\n");
                    break;
                case TIR.NTT.Qwen3MoE qwen3MoE:
                    IndentScope.Writer.IndWrite($"qwen3_moe({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {VisitBuffer(args[4], local: true).Name}, {VisitBuffer(args[5], local: true).Name}, {VisitBuffer(args[6], local: true).Name}, {VisitBuffer(args[7], local: true).Name}, {VisitBuffer(args[8], local: true).Name},{VisitBuffer(args[9], local: true).Name},{VisitBuffer(args[10], local: true).Name},{VisitBuffer(args[11], local: true).Name}, {qwen3MoE.LayerId}, {qwen3MoE.HiddenSize}, {qwen3MoE.IntermediateSize}, {qwen3MoE.MoEIntermediateSize}, {qwen3MoE.NumExpert}, {qwen3MoE.NumTopK}, {qwen3MoE.IsNormTopkProb});\n");
                    break;
                case TIR.NTT.SparseExperts sparseExperts:
                    IndentScope.Writer.IndWrite($"sparse_experts({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {VisitBuffer(args[4], local: true).Name}, {VisitBuffer(args[5], local: true).Name}, {VisitBuffer(args[6], local: true).Name}, {VisitBuffer(args[7], local: true).Name}, {VisitBuffer(args[8], local: true).Name}, {VisitBuffer(args[9], local: true).Name}, {VisitBuffer(args[10], local: true).Name}, {VisitBuffer(args[11], local: true).Name},{VisitBuffer(args[13], local: true).Name}, {sparseExperts.HiddenSize}, {sparseExperts.MoEIntermediateSize}, {sparseExperts.NumExpert}, {sparseExperts.NumTopK}, {sparseExperts.ChunkSize});\n");
                    break;
                case TIR.NTT.SparseExpertsGateUp gateUp:
                    IndentScope.Writer.IndWrite($"sparse_experts_gate_up({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {VisitBuffer(args[4], local: true).Name}, {VisitBuffer(args[5], local: true).Name}, {VisitBuffer(args[6], local: true).Name}, {VisitBuffer(args[7], local: true).Name}, {VisitBuffer(args[8], local: true).Name}, {gateUp.HiddenSize}, {gateUp.MoEIntermediateSize}, {gateUp.NumExpert}, {gateUp.NumTopK}, {gateUp.ChunkSize});\n");
                    break;
                case TIR.NTT.SparseExpertsDown down:
                    IndentScope.Writer.IndWrite($"sparse_experts_down({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {VisitBuffer(args[4], local: true).Name}, {VisitBuffer(args[5], local: true).Name}, {VisitBuffer(args[6], local: true).Name}, {down.HiddenSize}, {down.MoEIntermediateSize}, {down.NumExpert}, {down.NumTopK}, {down.ChunkSize});\n");
                    break;
                case TIR.NTT.TopK topK:
                    {
                        var inputBuffer = VisitBuffer(args[0], local: true);
                        var kOperand = args[1] is TIR.Buffer kBuffer
                            ? VisitBuffer(kBuffer, local: true)
                            : Visit(args[1]);
                        var outputBuffers = FlattenTuple(args[2]).Select(e => VisitBuffer(e, local: true)).ToArray();
                        if (outputBuffers.Length != 2)
                        {
                            throw new NotSupportedException("TopK expects two output buffers");
                        }

                        IndentScope.Writer.IndWrite($"top_k({inputBuffer.Name}, {kOperand.Name}, {outputBuffers[0].Name}, {outputBuffers[1].Name}, fixed_dim_v<{topK.Axis}>, {topK.Largest}, {topK.Sorted});\n");
                    }

                    break;
                default:
                    throw new NotSupportedException(kop.ToString());
            }
#if PROFILE_CALL
            IndentScope.Writer.Write($"printf(\"{expr.Target.GetType().Name} cost: %f\\n\", get_ms_time() - start_{CallCount++});\n");
#endif
        }
        else if (expr.Target is PrimFunction deviceFunc)
        {
            var parameters = deviceFunc.Parameters.ToArray();
            var arguments = expr.Arguments.ToArray();
            if (parameters.Length != arguments.Length)
            {
                throw new InvalidOperationException(
                    $"NTT call to {deviceFunc.Name} expects {parameters.Length} arguments, got {arguments.Length}.");
            }

            foreach (var item in arguments.OfType<TIR.Buffer>())
            {
                Visit(item);
            }
#if DEBUG_PRINT
            IndentScope.Writer.IndWrite($"runtime_util->printf(\"call {deviceFunc.Name} bid %d tid %d\\n\", bid, tid);\n");
#endif
            var argumentNames = new List<string>();
            var workspaceArguments = new Dictionary<MemoryLocation, BaseExpr>();
            for (int i = 0; i < parameters.Length; i++)
            {
                var parameter = parameters[i];
                var arg = arguments[i];
                if (parameter is BufferVar { Role: BufferVarRole.Workspace } workspace)
                {
                    if (!workspaceArguments.TryAdd(workspace.Location, arg))
                    {
                        throw new InvalidOperationException(
                            $"NTT call to {deviceFunc.Name} has duplicate {workspace.Location} workspace arguments.");
                    }

                    continue;
                }

                if (arg is TIR.Buffer b)
                {
                    var buffer = VisitBuffer(b, local: true);
                    argumentNames.Add(buffer.Name);
                }
                else
                {
                    argumentNames.Add(Visit(arg).Name);
                }
            }

            argumentNames.Add("rdata");
            argumentNames.Add("block_local_rdata");
            argumentNames.Add(GetWorkspacePointer(deviceFunc, workspaceArguments, MemoryLocation.Data));
            argumentNames.Add(GetWorkspacePointer(deviceFunc, workspaceArguments, MemoryLocation.BlockLocalData));
            argumentNames.Add("output");
            argumentNames.Add("nullptr");

            _refFuncs.Add(deviceFunc);
            WriteIndWithProfiler($"{deviceFunc.Name}({string.Join(", ", argumentNames)});\n");
        }
        else
        {
            var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
            switch (expr.Target)
            {
                case IR.Math.Binary op:
                    str = CSourceUtilities.ConvertBinary(op, arguments);
                    break;
                case IR.Math.Unary op:
                    str = CSourceUtilities.ConvertUnary(op, arguments);
                    break;
                case IR.Math.Compare op:
                    str = CSourceUtilities.ConvertCompare(op, arguments);
                    break;
                case IR.Math.Select op:
                    str = CSourceUtilities.ConvertSelect(op, arguments);
                    break;
                case TIR.Load op:
                    str = $"{arguments[0].Name}[{arguments[1].Name}]";
                    break;
                case TIR.Store op:
                    IndentScope.Writer.IndWrite($"{arguments[0].Name}[{arguments[1].Name}] = {arguments[1].Name};\n");
                    break;
                case TIR.NTT.PtrOf op:
                    str = op.PtrName;
                    break;
                case IR.Math.Clamp op:
                    str = CSourceUtilities.ConvertClamp(op, arguments);
                    break;
                case IR.Shapes.AsTensor op:
                    str = CSourceUtilities.ConvertAsTensor(op, arguments);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported call target: {expr.Target}");
            }
        }

        symbol = new(type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitConst(Const expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        string type;
        string str;
        if (expr is TensorConst { Value: Tensor { ElementType: PrimType ptype, Shape: { IsScalar: true } } scalar })
        {
            str = scalar[Array.Empty<long>()].ToString() switch
            {
                "True" => "1",
                "False" => "0",
                null => string.Empty,
                var x => x,
            };

            type = ptype.ToC();
        }
        else if (expr is TensorConst { Value: Tensor { ElementType: PointerType { ElemType: DataType }, Shape: { IsScalar: true } } pointer })
        {
            str = pointer.ToScalar<ulong>().ToString();
            type = "uint8_t *";
        }
        else
        {
            throw new NotSupportedException($"NTT C codegen does not support constant {expr} with type {expr.CheckedType}.");
        }

        symbol = new(type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitIfThenElse(IfThenElse expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var condition = Visit(expr.Condition);
        IndentScope.Writer.IndWrite($"if ({condition.Name}) {{\n");
        using (new IndentScope())
        {
            Visit(expr.Then);
        }

        if (expr.Else.Count > 0)
        {
            IndentScope.Writer.IndWrite("} else {\n");
            using (new IndentScope())
            {
                Visit(expr.Else);
            }
        }

        IndentScope.Writer.IndWrite("}\n");

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitLet(Let expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var var = Visit(expr.Var);
        var value = Visit(expr.Expression);
        IndentScope.Writer.IndWrite($"{var.Type} {var.Name} = {value.Name};\n");
        using (new IndentScope())
        {
            Visit(expr.Body);
        }

        var body = Visit(expr.Body);
        symbol = new(body.Type, body.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitSequential(Sequential expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        foreach (var field in expr.Fields)
        {
            if (field is Call call)
            {
                var name = Visit(call).Name;
                if (call.Target is not IR.Shapes.AsTensor)
                {
                    IndentScope.Writer.IndWrite(name);
                }
            }
            else
            {
                Visit(field);
            }
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitReturn(Return expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var returnValues = expr.Values.ToArray();
        if (returnValues.Length == 0)
        {
            symbol = new(string.Empty, string.Empty);
            _exprMemo.Add(expr, symbol);
            return symbol;
        }

        var outputParameters = VisitEntry.GetAbiView().OutputParameters;
        var resultStorages = returnValues
            .Select((value, resultIndex) => PrimFunctionAbi.GetResultStorage(VisitEntry, resultIndex, value))
            .ToArray();
        if (outputParameters.Count > 0)
        {
            IndentScope.Writer.IndWrite("if (output_descs != nullptr) {\n");
        }

        IndentScope? outputDescriptorScope = outputParameters.Count > 0 ? new IndentScope() : null;
        for (int outputIndex = 0; outputIndex < outputParameters.Count; outputIndex++)
        {
            var outputParameter = outputParameters[outputIndex];
            var matchingResultIndexes = resultStorages
                .Select((storage, resultIndex) => (Storage: storage, ResultIndex: resultIndex))
                .Where(binding => ReferenceEquals(binding.Storage, outputParameter))
                .Select(binding => binding.ResultIndex)
                .ToArray();
            if (matchingResultIndexes.Length != 1)
            {
                throw new InvalidOperationException(
                    $"NTT PrimFunction {VisitEntry.Name} Return must bind output parameter {outputParameter.Name} exactly once, " +
                    $"but found {matchingResultIndexes.Length} bindings.");
            }

            var resultValue = returnValues[matchingResultIndexes[0]];
            var value = Visit(resultValue);
            var rank = resultValue.CheckedShape.Rank;
            var physicalValue = resultValue is TIR.Buffer { DistributedType: not null } ||
                resultValue.CheckedType is DistributedType
                ? $"{value.Name}.local()"
                : value.Name;
            IndentScope.Writer.IndWrite($"output_descs[{outputIndex}].data = (std::byte *){physicalValue}.elements().data();\n");
            IndentScope.Writer.IndWrite($"output_descs[{outputIndex}].size = {physicalValue}.size() * sizeof(*{physicalValue}.elements().data());\n");
            IndentScope.Writer.IndWrite($"{value.Name}.shape().copy_to(output_descs[{outputIndex}].shape);\n");
            IndentScope.Writer.IndWrite($"{physicalValue}.strides().copy_to(output_descs[{outputIndex}].strides);\n");
            IndentScope.Writer.IndWrite($"output_descs[{outputIndex}].rank = {rank};\n");
        }

        outputDescriptorScope?.Dispose();
        if (outputParameters.Count > 0)
        {
            IndentScope.Writer.IndWrite("}\n");
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    private static IReadOnlyList<BaseExpr> FlattenTuple(BaseExpr expr)
    {
        var result = new List<BaseExpr>();
        var stack = new Stack<BaseExpr>();
        stack.Push(expr);
        while (stack.Count > 0)
        {
            var current = stack.Pop();
            if (current is IR.Tuple tuple)
            {
                for (int i = tuple.Count - 1; i >= 0; i--)
                {
                    stack.Push(tuple[i]);
                }
            }
            else
            {
                result.Add(current);
            }
        }

        return result;
    }

    private CSymbol VisitBuffer(BaseExpr buffer, bool local)
    {
        var symbol = Visit(buffer);
        if (local && ((buffer.CheckedType is DistributedType) || (buffer is TIR.Buffer b && b.DistributedType != null)))
        {
            return new CSymbol(symbol.Type, $"{symbol.Name}.local()");
        }

        return symbol;
    }

    private string VisitScalarTensor(BaseExpr expr)
    {
        var tensorType = expr.CheckedType switch
        {
            TensorType tt => tt,
            DistributedType dt => dt.TensorType,
            _ => throw new NotSupportedException($"NTT codegen expects scalar tensor argument, got {expr.CheckedType}."),
        };

        if (!tensorType.Shape.IsScalar)
        {
            throw new NotSupportedException($"NTT codegen expects scalar tensor argument, got shape {tensorType.Shape}.");
        }

        return $"{VisitBuffer(expr, local: true).Name}()";
    }

    private static void ValidateMatMulGluScales(IReadOnlyList<BaseExpr> args)
    {
        if (args.Count < 10 || args.Skip(5).Take(4).Any(arg => arg is not None))
        {
            throw new NotSupportedException("NTT MatMulGlu codegen currently supports only None input/weight scales.");
        }
    }

    private void DeclBuffer(TIR.Buffer buffer, CSymbol symbol)
    {
        if (_declaredBuffers.Add(buffer))
        {
            IndentScope.Writer.IndWrite($"auto {symbol.Name}");
            if (buffer.MemSpan.Buffer.Start is not None)
            {
                // If the buffer has a start, we create a tensor view
                var isReadOnly = buffer.ElemType is not ReferenceType &&
                    (buffer.MemSpan.Buffer.Location is MemoryLocation.Rdata or MemoryLocation.ChipLocalRdata or MemoryLocation.BlockLocalRdata ||
                     (buffer.MemSpan.Buffer.Location == MemoryLocation.Input && buffer.MemSpan.Buffer.Start is not BufferVar { Role: BufferVarRole.InOut }));
                var elementType = GetBufferElementType(buffer);
                var dtypeStr = isReadOnly ? $"const {elementType}" : elementType;
                var dimensions = buffer.DistributedType is null ? buffer.Dimensions : ((RankedShape)buffer.DistributedType.TensorType.Shape).Dimensions;
                var spanStr = $"span_cast<{dtypeStr}>({Visit(buffer.MemSpan).Name})";
                var dimensionValues = dimensions.AsValueEnumerable().Select(x => Visit(x).Name);
                var strideValues = buffer.Strides.AsValueEnumerable().Select(x => Visit(x).Name);

                if (buffer.DistributedType is DistributedType distributedType)
                {
                    var viewArguments = buffer.MemSpan.Buffer.Location switch
                    {
                        MemoryLocation.Rdata or MemoryLocation.ChipLocalRdata
                            => $"make_sharded_tensor_view_from_global_buffer({spanStr}",
                        MemoryLocation.Input or MemoryLocation.Output
                            => $"make_sharded_tensor_view_from_address({spanStr}.data()",
                        _ => $"make_sharded_tensor_view({spanStr}",
                    };
                    IndentScope.Writer.IndWrite($"= {viewArguments}, make_shape({StringUtility.Join(", ", dimensionValues)}), {KernelUtility.ShardingToC(distributedType)}, make_strides({StringUtility.Join(", ", strideValues)}))");
                }
                else
                {
                    IndentScope.Writer.IndWrite($"= make_tensor_view({spanStr}, make_shape({StringUtility.Join(", ", dimensionValues)}), make_strides({StringUtility.Join(", ", strideValues)}))");
                }
            }

            IndentScope.Writer.Write($";\n");
        }
    }

    private string AllocateLocalName(string preferredName)
    {
        if (_localNames.Add(preferredName))
        {
            return preferredName;
        }

        var suffix = _localNameSuffixes.TryGetValue(preferredName, out var nextSuffix)
            ? nextSuffix
            : 1;
        string candidate;
        do
        {
            candidate = $"{preferredName}_{suffix}";
            suffix++;
        }
        while (!_localNames.Add(candidate));

        _localNameSuffixes[preferredName] = suffix;
        return candidate;
    }

    private string GetBufferElementType(TIR.Buffer buffer)
    {
        if (buffer.ElemType is not ReferenceType)
        {
            return buffer.ElemType.ToC();
        }

        if (buffer.MemSpan.Buffer.Start is not IVar parameter)
        {
            throw new InvalidOperationException(
                $"NTT reference buffer {buffer.Name} must be backed by a runtime ABI parameter.");
        }

        var parameterIndex = Array.IndexOf(TensorParams, parameter);
        if (parameterIndex < 0)
        {
            throw new InvalidOperationException(
                $"NTT reference buffer {buffer.Name} backing parameter {parameter.Name} is not in the runtime ABI.");
        }

        return $"typename std::remove_cvref_t<T{parameterIndex}>::element_type";
    }

    private string GetWorkspacePointer(
        PrimFunction callee,
        IReadOnlyDictionary<MemoryLocation, BaseExpr> workspaceArguments,
        MemoryLocation location)
    {
        if (!workspaceArguments.TryGetValue(location, out var argument) || argument is not TIR.Buffer buffer)
        {
            throw new InvalidOperationException(
                $"NTT call to {callee.Name} requires a bufferized {location} workspace argument.");
        }

        var view = VisitBuffer(buffer, local: true);
        return $"reinterpret_cast<std::byte *>({view.Name}.elements().data())";
    }
}
