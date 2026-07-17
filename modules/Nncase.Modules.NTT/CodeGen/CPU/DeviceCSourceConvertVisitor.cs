// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define MULTI_CORE_XPU

// #define DEBUG_PRINT
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using DryIoc;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

public class DeviceCSourceConvertVisitor : CSourceConvertVisitor
{
    protected readonly StringBuilder _deviceBuilder;

    private readonly Dictionary<IVar, BaseExpr> _letBindings = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<int, string> _cacheArenaNames = new();
    private readonly Stack<Dictionary<Call, ReductionState>> _reductionScopes = new();
    private int _regionCopyCounter;
    private int _reductionStateCounter;

    public DeviceCSourceConvertVisitor(NTTTargetOptions targetOptions)
    {
        _deviceBuilder = new();
        TargetOptions = targetOptions;
    }

    public NTTTargetOptions TargetOptions { get; }

    public static void WriteWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal);
            if (index != -1)
            {
                tagName = functionName.Substring(0, index);
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
#if false // Disable device profiling for now.
        IndentScope.Writer.Write($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.Write($"profile_scope profiler(function_name, profile_level::device);\n");
#endif
        IndentScope.Writer.Write($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

    public static void WriteIndWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal);
            if (index != -1)
            {
                tagName = functionName.Substring(0, index);
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
#if false // Disable device profiling for now.
        IndentScope.Writer.IndWrite($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.IndWrite($"profile_scope profiler(function_name, profile_level::device);\n");
#endif
        IndentScope.Writer.IndWrite($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

    public string GetHeader()
    {
        return _deviceBuilder.ToString();
    }

    /// <inheritdoc/>
    protected override CSymbol VisitPrimFunction(PrimFunction expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        if (expr.CheckedType is not CallableType { ReturnType: TupleType r } || r != TupleType.Void)
        {
            throw new NotSupportedException("The PrimFunction must return void!");
        }

        var ctype = $"template<{string.Join(", ", Enumerable.Range(0, expr.Parameters.Length).Select(x => $"class T{x}"))}>" + Environment.NewLine +
            $"NTT_DEVICE void {expr.Name}({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit).Select((s, i) => $"T{i} &&{s.Name}").ToArray())})";

        using (var scope = new IndentScope(_deviceBuilder))
        {
            // 1. Function signature
            IndentScope.Writer.IndWrite($"{ctype} {{\n");

            // 2. Function body
            using (_ = new IndentScope())
            {
                foreach (var arena in CollectCacheArenas(expr))
                {
                    var name = $"cache_l{arena.Hierarchy}";
                    _cacheArenaNames.Add(arena.Hierarchy, name);
                    IndentScope.Writer.IndWrite($"alignas({arena.Alignment}) std::byte {name}[{arena.Size}];\n");
                }

                Visit(expr.Body);
            }

            // 3. Function closing
            IndentScope.Writer.IndWrite("}\n");
        }

        symbol = new(ctype, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitIfThenElse(IfThenElse expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var cond = Visit(expr.Condition);
        IndentScope.Writer.IndWrite($"if ({cond.Name}) {{\n");
        using (_ = new IndentScope())
        {
            Visit(expr.Then);
        }

        IndentScope.Writer.IndWrite("}\n");
        IndentScope.Writer.IndWrite("else {\n");
        using (_ = new IndentScope())
        {
            Visit(expr.Else);
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

        var @var = Visit(expr.Var);
        var value = Visit(expr.Expression);
        _exprMemo[(BaseExpr)expr.Var] = new(value.Type, @var.Name);
        var hadPreviousBinding = _letBindings.TryGetValue(expr.Var, out var previousBinding);
        _letBindings[expr.Var] = expr.Expression;

#if DEBUG_PRINT
        IndentScope.Writer.IndWrite($"runtime_util->printf(\"let {@var.Name}\\n\");\n");
#endif
        if (value.Type.StartsWith("array"))
        {
            var ss = value.Type.Split(" ");
            IndentScope.Writer.IndWrite($"{ss[1]} {@var.Name}[{ss[2]}];\n");
        }
        else
        {
            IndentScope.Writer.IndWrite($"{value.Type} {@var.Name} = {value.Name};\n");
        }

        try
        {
            Visit(expr.Body);
        }
        finally
        {
            if (hadPreviousBinding)
            {
                _letBindings[expr.Var] = previousBinding!;
            }
            else
            {
                _letBindings.Remove(expr.Var);
            }
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitPhysicalBuffer(PhysicalBuffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var start = Visit(expr.Start);
        var size = Visit(expr.Size);
        string name = expr.Location switch
        {
            MemoryLocation.Cache => _cacheArenaNames.TryGetValue(expr.Hierarchy, out var cacheArenaName)
                ? cacheArenaName
                : throw new InvalidOperationException($"Cache hierarchy {expr.Hierarchy} has no function-local arena."),
            MemoryLocation.Input or MemoryLocation.Output => start.Name,
            _ => throw new NotSupportedException(expr.Location.ToString()),
        };

        var str = $"ntt::span<std::byte, {size.Name}>({name} + {start.Name}, {size.Name})";
        symbol = new(start.Type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    private (int Hierarchy, int Alignment, long Size)[] CollectCacheArenas(PrimFunction function)
    {
        return ExprCollector.Collect(function.Body)
            .OfType<PhysicalBuffer>()
            .Where(buffer => buffer.Location == MemoryLocation.Cache)
            .GroupBy(buffer => buffer.Hierarchy)
            .Select(group =>
            {
                var space = TargetOptions.TargetMachineModel.TilingMemorySpaces.SingleOrDefault(
                    candidate => candidate.TIRBinding is { Location: MemoryLocation.Cache } binding &&
                        binding.Hierarchy == group.Key)
                    ?? throw new InvalidOperationException(
                        $"Target {TargetOptions.TargetMachineModel.Id} does not bind cache hierarchy {group.Key}.");
                var resource = TargetOptions.TargetMachineModel.GetMemoryResource(space);
                var size = group.Max(buffer => checked(GetFixedBufferStart(buffer) + buffer.Size.FixedValue));
                if (size <= 0 || size > space.MaxAllocationBytesPerScope)
                {
                    throw new InvalidOperationException(
                        $"Cache hierarchy {group.Key} requires {size} bytes, outside its allocation limit {space.MaxAllocationBytesPerScope}.");
                }

                var alignment = Math.Max(resource.AllocationGranularityBytes, group.Max(buffer => buffer.Alignment));
                return (group.Key, alignment, size);
            })
            .OrderBy(arena => arena.Key)
            .Select(arena => (arena.Key, arena.alignment, arena.size))
            .ToArray();
    }

    private static long GetFixedBufferStart(PhysicalBuffer buffer)
    {
        if (!buffer.Size.IsFixed)
        {
            throw new InvalidOperationException(
                $"Cache buffer allocation size must be fixed before NTT codegen, got {buffer.Size}.");
        }

        try
        {
            return buffer.Start switch
            {
                TensorConst { Value: Tensor { ElementType: PointerType, Shape.IsScalar: true } pointer } =>
                    checked((long)pointer.ToScalar<ulong>()),
                TensorConst { Value.Shape.IsScalar: true } tensorConst =>
                    Convert.ToInt64(tensorConst.Value[Array.Empty<long>()]),
                _ => throw new InvalidOperationException(
                    $"Cache buffer start must be allocated before NTT codegen, got {buffer.Start}."),
            };
        }
        catch (OverflowException ex)
        {
            throw new InvalidOperationException($"Cache buffer start is outside Int64 range: {buffer.Start}.", ex);
        }
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
        var size = Visit(expr.Size);

        var str = $"make_subspan({buffer.Name}, {start.Name}, {size.Name})";
        symbol = new(start.Type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitBuffer(TIR.Buffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var dimensions = expr.Dimensions;
        var isFixedDimensions = dimensions.AsValueEnumerable().All(x => x.IsFixed);
        var isFixedStrides = expr.Strides.AsValueEnumerable().All(x => x.IsFixed);
        var dimensionSymbols = dimensions.AsValueEnumerable().Select(Visit).ToArray();
        var strideSymbols = expr.Strides.AsValueEnumerable().Select(Visit).ToArray();

        var dtypeStr = expr.ElemType.ToC();
        var dimensionStr = KernelUtility.DimensionsToC(isFixedDimensions, dimensionSymbols, true);
        var strideStr = KernelUtility.StridesToC(isFixedStrides, strideSymbols, true);
        var type = $"tensor_view<{dtypeStr}, {dimensionStr}, {strideStr}> ";

        symbol = new(type, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitCall(Call expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        string type = expr.CheckedType switch
        {
            TupleType x when x == TupleType.Void => string.Empty,
            TensorType { IsScalar: true } x => x.DType switch
            {
                ReferenceType => "auto",
                _ => x.DType.ToC(),
            },
            TensorType or DistributedType or DimensionType => "auto",
            _ => throw new NotSupportedException(),
        };

        string str = string.Empty;
        var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        if (TryGetReductionState(expr, out var reductionState))
        {
            EmitReductionKernel(reductionState, arguments, ReductionKernelPhase.Accumulate);
            reductionState.UpdateCount++;
            symbol = new(type, str);
            _exprMemo.Add(expr, symbol);
            return symbol;
        }

        switch (expr.Target)
        {
            case PrimFunction deviceFunc:
                WriteIndWithProfiler($"{deviceFunc.Name}({string.Join(",", arguments.Select(arg => arg.Name))});\n");
                break;
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
            case IR.Shapes.AsTensor op:
                str = arguments[0].Name;
                break;
            case IR.Tensors.LocalShardDim op:
                str = $"local_shard_dim<0>(make_sharding<{op.Placement.PlacementToC()}>({op.AxisPolicy.SBPToC()}), make_shape({arguments[0].Name}))";
                break;
            case TIR.NTT.SramPtr op:
                str = $"g_cpu_mt->sram_address(bid, tid) + {arguments[0].Name}";
                break;
            case TIR.Load op:
                str = $"{arguments[0].Name}[{arguments[1].Name}]";
                break;
            case TIR.Store op:
#if DEBUG_PRINT
                IndentScope.Writer.IndWrite($"runtime_util->printf(\"{arguments[0].Name}[%d]\\n\", {arguments[1].Name});\n");
#endif
                IndentScope.Writer.IndWrite($"{arguments[0].Name}[{arguments[1].Name}] = {arguments[2].Name};\n");
                break;
            case TIR.NTT.PtrOf op:
                str = op.PtrName + ".data()";
                break;
            case IR.Buffers.Allocate op:
                if (op.Malloc)
                {
                    str = $"({type})runtime_util->malloc({arguments[0].Name})";
                }
                else
                {
                    type = $"array {((PointerType)expr.CheckedDataType).ElemType.ToC()} {arguments[0].Name}";
                    str = $"";
                }

                break;
            case IR.Buffers.BufferSubview op:
                {
                    var arg0 = VisitDimOrShape(expr.Arguments[1], CShapeKind.Shape).Name;
                    var arg1 = VisitDimOrShape(expr.Arguments[2], CShapeKind.Shape).Name;
                    str = $"{arguments[0].Name}.view({arg0}, {arg1})";
                }

                break;
            case IR.Buffers.AllocateBufferView op:
                {
                    var buffer = (TIR.Buffer)expr.Arguments[0];
                    var dimensions = buffer.Dimensions;
                    var isFixedDimensions = dimensions.AsValueEnumerable().All(x => x.IsFixed);
                    var isFixedStrides = buffer.Strides.AsValueEnumerable().All(x => x.IsFixed);
                    var dimensionSymbols = dimensions.AsValueEnumerable().Select(Visit).ToArray();
                    var strideSymbols = buffer.Strides.AsValueEnumerable().Select(Visit).ToArray();

                    var dtypeStr = buffer.ElemType.ToC();
                    var dimensionStrs = dimensionSymbols.Select(x => x.Name);
                    var strideStrs = strideSymbols.Select(x => x.Name);
                    str = $"make_tensor_view(span_cast<{dtypeStr}>({Visit(buffer.MemSpan).Name}), make_shape({StringUtility.Join(", ", dimensionStrs)}), make_strides({StringUtility.Join(", ", strideStrs)}))";
                }

                break;
            case IR.Tensors.Cast op:
                str = $"(({op.NewType.ToC()}){arguments[0].Name})";
                break;
            case TIR.Memcopy op:
                WriteIndWithProfiler($"tensor_copy_sync({arguments[1].Name}, {arguments[0].Name});\n");
                break;
            case TIR.NTT.TensorLoad:
                if (arguments.Length != 2)
                {
                    throw new NotSupportedException($"NTT device TensorLoad expects (destination, source), got {arguments.Length} operands.");
                }

                EmitTensorRegionLoad(expr.Arguments[0], expr.Arguments[1]);
                break;
            case TIR.NTT.TensorStore:
                if (arguments.Length != 2)
                {
                    throw new NotSupportedException($"NTT device TensorStore expects (source, destination), got {arguments.Length} operands.");
                }

                EmitTensorRegionStore(expr.Arguments[0], expr.Arguments[1]);
                break;
            case TIR.TileLoad:
                WriteIndWithProfiler($"tensor_copy_sync({arguments[1].Name}, {arguments[0].Name});\n");
                break;
            case TIR.TileStore:
                WriteIndWithProfiler($"tensor_copy_sync({arguments[0].Name}, {arguments[1].Name});\n");
                break;
            case TIR.NTT.Barrier barrier:
                WriteTopologyBarrier(barrier.Scope);
                break;
            case TIR.NTT.SynchronizeThreads:
                WriteTopologyBarrier(TIR.NTT.BarrierScope.Block);
                break;
            case TIR.NTT.Unary op:
                WriteIndWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    UnaryOp = op.UnaryOp,
                }).Result);
                break;
            case TIR.NTT.VectorizedBinary op:
                WriteIndWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    BinaryOp = op.BinaryOp,
                }).Result);
                break;
            case TIR.NTT.Swish swish:
                if (swish.Beta == 1.0f)
                {
                    WriteIndWithProfiler($"unary<ops::swish>({arguments[0].Name}, {arguments[1].Name});\n");
                }
                else
                {
                    IndentScope.Writer.IndWrite($"\n{{\nauto b= {swish.Beta}; auto tb = make_tensor_view_from_address<float>(&b, fixed_shape_v<>);\n");
                    WriteIndWithProfiler($"binary<ops::swishb>({arguments[0].Name}, tb, {arguments[1].Name});\n}}\n");
                }

                break;
            case TIR.NTT.Matmul matmul:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", new TypedKernelTemplateModel<TIR.NTT.Matmul>(matmul)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.PackedMatMul matmul:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedMatMul.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedMatMul>(matmul)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.QKVParallelLinear qkvParallelLinear:
                ValidateQKVParallelLinearScales(expr.Arguments.ToArray());
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/QKVParallelLinear.cshtml", new TypedKernelTemplateModel<TIR.NTT.QKVParallelLinear>(qkvParallelLinear)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.PackedQKVParallelLinear packedQKVParallelLinear:
                ValidateQKVParallelLinearScales(expr.Arguments.ToArray());
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedQKVParallelLinear.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedQKVParallelLinear>(packedQKVParallelLinear)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.MatMulGlu matMulGlu:
                ValidateMatMulGluScales(expr.Arguments.ToArray());
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/MatMulGlu.cshtml", new TypedKernelTemplateModel<TIR.NTT.MatMulGlu>(matMulGlu)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.PackedMatMulGlu packedMatMulGlu:
                ValidateMatMulGluScales(expr.Arguments.ToArray());
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedMatMulGlu.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedMatMulGlu>(packedMatMulGlu)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.Pack vectorize:
                WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Pack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Pack>(vectorize)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Transpose transpose:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Transpose.cshtml", new TypedKernelTemplateModel<TIR.NTT.Transpose>(transpose)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Unpack devectorize:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unpack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Unpack>(devectorize)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Reduce reduce:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Reduce.cshtml", new TypedKernelTemplateModel<TIR.NTT.Reduce>(reduce)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.RoPE rope:
                WriteIndWithProfiler($"rope({arguments[0].Name}, {arguments[1].Name}, {arguments[2].Name}, {arguments[3].Name});\n");
                break;
            case TIR.NTT.Cast cast:
                {
                    string postOps = string.Empty;
                    if (expr[TIR.NTT.Cast.PostOps] is Fusion lambda)
                    {
                        postOps = $"<{lambda.Name}>";
                    }

                    IndentScope.Writer.IndWrite($"cast{postOps}({arguments[0].Name}, {arguments[1].Name}, fixed_shape_v<{string.Join(",", cast.VectorizeAxes.ToArray())}>);\n");
                }

                break;
            case TIR.NTT.VectorizedLayerNorm lm:
                {
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/VectorizedLayerNorm.cshtml", new TypedKernelTemplateModel<TIR.NTT.VectorizedLayerNorm>(lm)
                    {
                        Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).Concat(lm.PadedNums.Select(Visit).Select(x => new KernelArgument { Symbol = x })).ToArray(),
                        Indent = new string(' ', IndentScope.Writer.Indent),
                        Args = expr.Arguments[..1].ToArray(),
                    }).Result);
                }

                break;
            case TIR.NTT.Pad pad:
                {
                    var padValueType = expr.Arguments[0].CheckedTensorType.DType is VectorType vt ? vt.ElemType : expr.Arguments[0].CheckedTensorType.DType;
                    WriteWithProfiler($"pad({arguments[0].Name}, {arguments[2].Name}, {arguments[1].Name}, {expr.Arguments[0].CheckedDataType.ToC()} {{ ({padValueType.ToC()}){pad.PadValue} }});\n");
                }

                break;
            case TIR.NTT.Where where:
                WriteWithProfiler($"where({arguments[0].Name}, {arguments[1].Name}, {arguments[2].Name}, {arguments[3].Name});\n");
                break;
            case TIR.NTT.GetPositionIds getPositionIds:
                WriteIndWithProfiler($"get_position_ids({arguments[0].Name}, {arguments[1].Name}, {KernelUtility.ShardingToC(getPositionIds.DistributedType)}, {Visit(getPositionIds.DistributedType.TensorType.Shape).Name});\n");
                break;
            case TIR.NTT.Compare compare:
                {
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Compare.cshtml", new CompareKernelTemplateModel
                    {
                        Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                        CompareOp = compare.CompareOp,
                    }).Result);
                }

                break;

            default:
                throw new NotSupportedException($"Unsupported call target: {expr.Target}");
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
            type = pointer.ElementType.ToC();
        }
        else
        {
            throw new NotSupportedException();
        }

        symbol = new(type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitTupleConst(TupleConst tp)
    {
        if (_exprMemo.TryGetValue(tp, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = $"{string.Join(",", tp.Value.Select(x => Visit(Const.FromValue(x)).Name))}";
        symbol = new(type, str);
        _exprMemo.Add(tp, symbol);
        return symbol;
    }

    protected override CSymbol VisitTuple(IR.Tuple tp)
    {
        if (_exprMemo.TryGetValue(tp, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = $"{string.Join(",", tp.Fields.AsValueEnumerable().Select(x => Visit(x).Name).ToArray())}";
        symbol = new(type, str);
        _exprMemo.Add(tp, symbol);
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
    protected override CSymbol VisitSequential(Sequential expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        for (var index = 0; index < expr.Fields.Length; index++)
        {
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

            Visit(expr.Fields[index]);
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitFor(For expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var ownsReductionScope = expr.Mode == LoopMode.Reduction && _reductionScopes.Count == 0;
        Dictionary<Call, ReductionState>? reductionScope = null;
        if (ownsReductionScope)
        {
            reductionScope = CreateReductionScope(expr);
            _reductionScopes.Push(reductionScope);
        }

        try
        {
            if (ownsReductionScope)
            {
                EmitReductionInitializers(reductionScope!);
            }

            // 1. For Loop signature
            var loopVar = Visit(expr.LoopVar);
            IndentScope.Writer.IndWrite($"for ({loopVar.Type} {loopVar.Name} = {Visit(expr.Domain.Start).Name}; {loopVar.Name} < {Visit(expr.Domain.Stop).Name}; {loopVar.Name} += {Visit(expr.Domain.Step).Name}) {{\n");
#if DEBUG_PRINT
            IndentScope.Writer.IndWrite($"runtime_util->printf(\"{loopVar.Name} = %d\\n\", {loopVar.Name});\n");
#endif

            using (_ = new IndentScope())
            {
                // 2. For Body
                Visit(expr.Body);
            }

            // 3. For closing
            IndentScope.Writer.IndWrite("}\n");

            if (ownsReductionScope)
            {
                EmitReductionFinalizers(reductionScope!);
            }
        }
        finally
        {
            if (ownsReductionScope)
            {
                _reductionScopes.Pop();
            }
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    private void VisitPartitionedReductionLoops(For fullLoop, For tailLoop)
    {
        var reductionScope = CreateReductionScope(fullLoop.Body, tailLoop.Body);
        _reductionScopes.Push(reductionScope);
        try
        {
            EmitReductionInitializers(reductionScope);
            Visit(fullLoop);
            Visit(tailLoop);
            EmitReductionFinalizers(reductionScope);
        }
        finally
        {
            _reductionScopes.Pop();
        }
    }

    private Dictionary<Call, ReductionState> CreateReductionScope(For reductionLoop)
        => CreateReductionScope(reductionLoop.Body);

    private Dictionary<Call, ReductionState> CreateReductionScope(params BaseExpr[] bodies)
    {
        var groups = ReductionCodegenUtility.CollectReductionCallGroups(bodies);
        if (groups.Length == 0)
        {
            throw new InvalidOperationException("NTT reduction scope contains no backend reduction operation.");
        }

        var scope = new Dictionary<Call, ReductionState>(ReferenceEqualityComparer.Instance);
        foreach (var group in groups)
        {
            var call = group.Prototype;
            var kind = call.Target switch
            {
                TIR.NTT.Matmul => ReductionKernelKind.Matmul,
                TIR.NTT.PackedMatMul => ReductionKernelKind.PackedMatmul,
                TIR.NTT.QKVParallelLinear => ReductionKernelKind.QKVParallelLinear,
                TIR.NTT.PackedQKVParallelLinear => ReductionKernelKind.PackedQKVParallelLinear,
                TIR.NTT.MatMulGlu => ReductionKernelKind.MatMulGlu,
                TIR.NTT.PackedMatMulGlu => ReductionKernelKind.PackedMatMulGlu,
                TIR.NTT.Reduce => ReductionKernelKind.Reduce,
                _ => throw new InvalidOperationException(
                    $"Unsupported NTT reduction operation {call.Target.GetType().Name}."),
            };
            var accumulatorOperands = ReductionCodegenUtility.GetAccumulatorOperands(call);
            var stateOutputs = kind switch
            {
                ReductionKernelKind.MatMulGlu or ReductionKernelKind.PackedMatMulGlu
                    when accumulatorOperands.Length == 1 =>
                    new[] { accumulatorOperands[0].Argument, accumulatorOperands[0].Argument },
                _ => accumulatorOperands.Select(operand => operand.Argument).ToArray(),
            };
            var expectedStateCount = kind switch
            {
                ReductionKernelKind.QKVParallelLinear or ReductionKernelKind.PackedQKVParallelLinear => 3,
                ReductionKernelKind.MatMulGlu or ReductionKernelKind.PackedMatMulGlu => 2,
                _ => 1,
            };
            if (stateOutputs.Length != expectedStateCount)
            {
                throw new InvalidOperationException(
                    $"NTT reduction operation {call.Target.GetType().Name} requires {expectedStateCount} backend accumulator states, got {stateOutputs.Length} from its operand contract.");
            }

            var stateId = _reductionStateCounter++;
            var elementCount = call.Target is TIR.NTT.Reduce { ReduceOp: ReduceOp.Mean }
                ? $"ntt_reduction_{stateId}_element_count"
                : null;
            var state = new ReductionState(
                call,
                kind,
                stateOutputs,
                Enumerable.Range(0, stateOutputs.Length)
                    .Select(index => $"ntt_reduction_{stateId}_acc{index}")
                    .ToArray(),
                $"ntt_reduction_{stateId}_initialized",
                elementCount,
                group.Calls.Length);
            foreach (var groupedCall in group.Calls)
            {
                scope.Add(groupedCall, state);
            }
        }

        return scope;
    }

    private void EmitReductionFinalizers(IReadOnlyDictionary<Call, ReductionState> scope)
    {
        foreach (var state in GetDistinctReductionStates(scope))
        {
            if (state.UpdateCount != state.ExpectedUpdateCount)
            {
                throw new InvalidOperationException(
                    $"NTT reduction operation {state.Call.Target.GetType().Name} emitted " +
                    $"{state.UpdateCount} of {state.ExpectedUpdateCount} expected accumulator updates.");
            }

            var arguments = state.Call.Arguments.AsValueEnumerable().Select(Visit).ToArray();
            EmitReductionKernel(state, arguments, ReductionKernelPhase.Finalize);
        }
    }

    private static IEnumerable<ReductionState> GetDistinctReductionStates(
        IReadOnlyDictionary<Call, ReductionState> scope)
        => new HashSet<ReductionState>(scope.Values, ReferenceEqualityComparer.Instance);

    private static void WriteTopologyBarrier(TIR.NTT.BarrierScope scope)
    {
        var topology = scope switch
        {
            TIR.NTT.BarrierScope.Block => "block",
            TIR.NTT.BarrierScope.Chip => "chip",
            _ => throw new ArgumentOutOfRangeException(nameof(scope), scope, null),
        };
        WriteIndWithProfiler($"ntt::distributed::topology_synchronize<ntt::distributed::topology::{topology}>();\n");
    }

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

    private static string GetAccumulatorScalarType(ReductionKernelKind kind, BaseExpr output)
    {
        if (kind is not ReductionKernelKind.Reduce)
        {
            return "float";
        }

        var scalarType = output.CheckedDataType switch
        {
            PrimType primType => primType,
            VectorType vectorType => vectorType.ElemType,
            MaskVectorType => DataTypes.Boolean,
            var dataType => throw new NotSupportedException(
                $"NTT Reduce accumulator does not support output dtype {dataType}."),
        };
        return scalarType.IsFloat() ? "float" : scalarType.ToC();
    }

    private static long GetAccumulatorExtent(Dimension dimension, string context)
    {
        if (dimension.IsFixed)
        {
            if (dimension.FixedValue < 1)
            {
                throw new NotSupportedException(
                    $"{context} requires a positive extent, got {dimension.FixedValue}.");
            }

            return dimension.FixedValue;
        }

        if (dimension.Metadata.Range is not { } range ||
            !double.IsFinite(range.Max) ||
            range.Max != Math.Truncate(range.Max) ||
            range.Max < 1 ||
            range.Max > long.MaxValue)
        {
            throw new NotSupportedException(
                $"{context} requires a positive finite integer maximum extent, got {dimension} with range {dimension.Metadata.Range}.");
        }

        return checked((long)range.Max);
    }

    private void EmitReductionInitializers(IReadOnlyDictionary<Call, ReductionState> scope)
    {
        foreach (var state in GetDistinctReductionStates(scope))
        {
            for (var index = 0; index < state.Outputs.Length; index++)
            {
                var output = state.Outputs[index];
                var outputSymbol = Visit(output).Name;
                var accumulator = state.Accumulators[index];
                var storage = $"{accumulator}_storage";
                var elementType = $"{accumulator}_element_t";
                var shape = output.CheckedShape
                    .Select((dimension, axis) => GetAccumulatorExtent(
                        dimension,
                        $"NTT {state.Call.Target.GetType().Name} accumulator {index} axis {axis}"))
                    .ToArray();
                var fixedShape = $"fixed_shape_v<{string.Join(",", shape)}>";
                var scalarType = GetAccumulatorScalarType(state.Kind, output);
                IndentScope.Writer.IndWrite($"using {elementType} = replace_element_t<typename std::decay_t<decltype({outputSymbol})>::element_type, {scalarType}>;\n");
                IndentScope.Writer.IndWrite($"auto {storage} = make_tensor<{elementType}>({fixedShape});\n");
                IndentScope.Writer.IndWrite($"auto {accumulator} = make_tensor_view_from_address<{elementType}>({storage}.elements().data(), {outputSymbol}.shape());\n");
            }

            IndentScope.Writer.IndWrite($"bool {state.Initialized} = false;\n");
            if (state.ElementCount is not null)
            {
                IndentScope.Writer.IndWrite($"size_t {state.ElementCount} = 0;\n");
            }
        }
    }

    private void EmitReductionKernel(
        ReductionState state,
        CSymbol[] arguments,
        ReductionKernelPhase phase)
    {
        if (phase == ReductionKernelPhase.Accumulate && state.UpdateCount >= state.ExpectedUpdateCount)
        {
            throw new InvalidOperationException(
                $"NTT reduction operation {state.Call.Target.GetType().Name} exceeds its " +
                $"{state.ExpectedUpdateCount} planned accumulator updates.");
        }

        var context = new ReductionKernelTemplateContext(
            phase,
            state.Accumulators,
            state.Initialized,
            state.ElementCount);
        if (state.Call.Target is TIR.NTT.QKVParallelLinear or TIR.NTT.PackedQKVParallelLinear)
        {
            ValidateQKVParallelLinearScales(state.Call.Arguments.ToArray());
        }
        else if (state.Call.Target is TIR.NTT.MatMulGlu or TIR.NTT.PackedMatMulGlu)
        {
            ValidateMatMulGluScales(state.Call.Arguments.ToArray());
        }

        var source = state.Call.Target switch
        {
            TIR.NTT.Matmul op => RenderReductionKernelTemplate(
                "~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", op, arguments, context),
            TIR.NTT.PackedMatMul op => RenderReductionKernelTemplate(
                "~/CodeGen/CPU/Templates/Kernels/PackedMatMul.cshtml", op, arguments, context),
            TIR.NTT.QKVParallelLinear op => RenderReductionKernelTemplate(
                "~/CodeGen/CPU/Templates/Kernels/QKVParallelLinear.cshtml", op, arguments, context),
            TIR.NTT.PackedQKVParallelLinear op => RenderReductionKernelTemplate(
                "~/CodeGen/CPU/Templates/Kernels/PackedQKVParallelLinear.cshtml", op, arguments, context),
            TIR.NTT.MatMulGlu op => RenderReductionKernelTemplate(
                "~/CodeGen/CPU/Templates/Kernels/MatMulGlu.cshtml", op, arguments, context),
            TIR.NTT.PackedMatMulGlu op => RenderReductionKernelTemplate(
                "~/CodeGen/CPU/Templates/Kernels/PackedMatMulGlu.cshtml", op, arguments, context),
            TIR.NTT.Reduce op => RenderReductionKernelTemplate(
                "~/CodeGen/CPU/Templates/Kernels/Reduce.cshtml", op, arguments, context),
            _ => throw new InvalidOperationException(
                $"Unsupported NTT reduction operation {state.Call.Target.GetType().Name}."),
        };
        IndentScope.Writer.Write(source);
    }

    private static string RenderReductionKernelTemplate<TOp>(
        string templatePath,
        TOp op,
        CSymbol[] arguments,
        ReductionKernelTemplateContext context)
        where TOp : Op =>
        RazorTemplateEngine.RenderAsync(templatePath, new TypedKernelTemplateModel<TOp>(op)
        {
            Arguments = arguments.Select(argument => new KernelArgument { Symbol = argument }).ToArray(),
            Indent = new string(' ', IndentScope.Writer.Indent),
            Reduction = context,
        }).Result;

    private void EmitTensorRegionLoad(BaseExpr destination, BaseExpr source)
    {
        var dest = GetBufferViewDescriptor(destination, "TensorLoad destination");
        var src = GetBufferViewDescriptor(source, "TensorLoad source");
        ValidateRegionCopyDescriptors("TensorLoad", src, dest);
        var sourceOffsets = BuildRelativeOffsets(dest.GlobalOffsets, src.GlobalOffsets);
        var offsetName = $"ntt_region_copy_{_regionCopyCounter++}_offset";
        IndentScope.Writer.IndWrite("{\n");
        using (_ = new IndentScope())
        {
            IndentScope.Writer.IndWrite($"auto {offsetName} = make_shape({string.Join(", ", sourceOffsets)});\n");
            IndentScope.Writer.IndWrite($"tensor_copy_sync({src.Symbol}.view({offsetName}, {dest.Symbol}.shape()), {dest.Symbol});\n");
        }

        IndentScope.Writer.IndWrite("}\n");
    }

    private void EmitTensorRegionStore(BaseExpr source, BaseExpr destination)
    {
        var src = GetBufferViewDescriptor(source, "TensorStore source");
        var dest = GetBufferViewDescriptor(destination, "TensorStore destination");
        ValidateRegionCopyDescriptors("TensorStore", src, dest);
        var destinationOffsets = BuildRelativeOffsets(src.GlobalOffsets, dest.GlobalOffsets);
        var offsetName = $"ntt_region_copy_{_regionCopyCounter++}_offset";
        IndentScope.Writer.IndWrite("{\n");
        using (_ = new IndentScope())
        {
            IndentScope.Writer.IndWrite($"auto {offsetName} = make_shape({string.Join(", ", destinationOffsets)});\n");
            IndentScope.Writer.IndWrite($"tensor_copy_sync({src.Symbol}, {dest.Symbol}.view({offsetName}, {src.Symbol}.shape()));\n");
        }

        IndentScope.Writer.IndWrite("}\n");
    }

    private BufferViewDescriptor GetBufferViewDescriptor(BaseExpr expression, string context)
    {
        var symbol = Visit(expression).Name;
        var resolved = ResolveBoundExpression(expression, context);
        switch (resolved)
        {
            case Call { Target: IR.Buffers.AllocateBufferView } allocate:
                {
                    var args = allocate.Arguments.ToArray();
                    if (args.Length != 2 || args[0] is not TIR.Buffer buffer)
                    {
                        throw new NotSupportedException(
                            $"NTT {context} expects AllocateBufferView(buffer, offsets).");
                    }

                    var localOffsets = GetShapeDimensions(args[1], $"{context} AllocateBufferView offsets");
                    if (localOffsets.Length != buffer.Rank)
                    {
                        throw new NotSupportedException(
                            $"NTT {context} AllocateBufferView rank mismatch: buffer rank {buffer.Rank}, offsets rank {localOffsets.Length}.");
                    }

                    return new BufferViewDescriptor(
                        symbol,
                        buffer.ElemType,
                        buffer.Dimensions.AsValueEnumerable().Select(dimension => Visit(dimension).Name).ToArray(),
                        AddOffsets(BuildDistributedGlobalOffsets(buffer.DistributedType, buffer.Rank), localOffsets));
                }
            case Call { Target: IR.Buffers.BufferSubview } subview:
                {
                    var args = subview.Arguments.ToArray();
                    if (args.Length != 3)
                    {
                        throw new NotSupportedException(
                            $"NTT {context} expects BufferSubview(buffer, offsets, shape).");
                    }

                    var source = GetBufferViewDescriptor(args[0], $"{context} subview source");
                    var localOffsets = GetShapeDimensions(args[1], $"{context} subview offsets");
                    var shape = GetShapeDimensions(args[2], $"{context} subview shape");
                    if (localOffsets.Length != source.Rank || shape.Length != source.Rank)
                    {
                        throw new NotSupportedException(
                            $"NTT {context} BufferSubview rank mismatch: source rank {source.Rank}, " +
                            $"offsets rank {localOffsets.Length}, shape rank {shape.Length}.");
                    }

                    return new BufferViewDescriptor(
                        symbol,
                        source.ElemType,
                        shape.Select(dimension => Visit(dimension).Name).ToArray(),
                        AddOffsets(source.GlobalOffsets, localOffsets));
                }
            case TIR.Buffer buffer:
                return new BufferViewDescriptor(
                    symbol,
                    buffer.ElemType,
                    buffer.Dimensions.AsValueEnumerable().Select(dimension => Visit(dimension).Name).ToArray(),
                    BuildDistributedGlobalOffsets(buffer.DistributedType, buffer.Rank));
            case IVar variable:
                {
                    var tensorType = variable.CheckedType switch
                    {
                        TensorType type => type,
                        DistributedType type => type.TensorType,
                        var type => throw new NotSupportedException(
                            $"NTT {context} expects a tensor buffer variable, got {type}."),
                    };
                    var distributedType = variable.CheckedType as DistributedType;
                    var dimensions = ((RankedShape)tensorType.Shape).Dimensions;
                    return new BufferViewDescriptor(
                        symbol,
                        tensorType.DType,
                        dimensions.AsValueEnumerable().Select(dimension => Visit(dimension).Name).ToArray(),
                        BuildDistributedGlobalOffsets(distributedType, tensorType.Shape.Rank));
                }
            default:
                throw new NotSupportedException(
                    $"NTT {context} cannot resolve buffer expression {resolved.GetType().Name}.");
        }
    }

    private BaseExpr ResolveBoundExpression(BaseExpr expression, string context)
    {
        var visited = new HashSet<IVar>(ReferenceEqualityComparer.Instance);
        while (expression is IVar variable && _letBindings.TryGetValue(variable, out var binding))
        {
            if (!visited.Add(variable))
            {
                throw new InvalidOperationException($"NTT {context} contains a cyclic Let binding at {variable.Name}.");
            }

            expression = binding;
        }

        return expression;
    }

    private string[] BuildDistributedGlobalOffsets(DistributedType? distributedType, int rank)
    {
        if (distributedType is null)
        {
            return Enumerable.Repeat("0_dim", rank).ToArray();
        }

        if (distributedType.TensorType.Shape.Rank != rank)
        {
            throw new InvalidOperationException(
                $"NTT distributed buffer rank mismatch: tensor rank {distributedType.TensorType.Shape.Rank}, buffer rank {rank}.");
        }

        var globalDimensions = ((RankedShape)distributedType.TensorType.Shape).Dimensions;
        var globalShape = $"make_shape({string.Join(", ", globalDimensions.AsValueEnumerable().Select(dimension => Visit(dimension).Name).ToArray())})";
        var sharding = KernelUtility.ShardingToC(distributedType);
        var mesh = distributedType.Placement.PlacementToC();
        var globalOffset = $"({sharding}).global_offset({globalShape}, {mesh}::local_index())";
        return Enumerable.Range(0, rank)
            .Select(axis => $"{globalOffset}[{axis}_dim]")
            .ToArray();
    }

    private string[] AddOffsets(IReadOnlyList<string> baseOffsets, IReadOnlyList<Dimension> localOffsets)
    {
        if (baseOffsets.Count != localOffsets.Count)
        {
            throw new InvalidOperationException(
                $"NTT buffer offset rank mismatch: base rank {baseOffsets.Count}, local rank {localOffsets.Count}.");
        }

        return baseOffsets.Zip(localOffsets)
            .Select(pair => pair.Second.IsFixed && pair.Second.FixedValue == 0
                ? pair.First
                : pair.First == "0_dim"
                    ? Visit(pair.Second).Name
                    : $"({pair.First} + {Visit(pair.Second).Name})")
            .ToArray();
    }

    private static string[] BuildRelativeOffsets(
        IReadOnlyList<string> regionOffsets,
        IReadOnlyList<string> containerOffsets)
    {
        if (regionOffsets.Count != containerOffsets.Count)
        {
            throw new InvalidOperationException(
                $"NTT region copy rank mismatch: region rank {regionOffsets.Count}, container rank {containerOffsets.Count}.");
        }

        return regionOffsets.Zip(containerOffsets)
            .Select(pair => pair.First == pair.Second
                ? "0_dim"
                : pair.Second == "0_dim"
                    ? pair.First
                    : $"({pair.First} - {pair.Second})")
            .ToArray();
    }

    private static Dimension[] GetShapeDimensions(BaseExpr expression, string context)
        => expression is RankedShape shape
            ? shape.Dimensions.ToArray()
            : throw new NotSupportedException(
                $"NTT {context} expects a ranked shape expression, got {expression.GetType().Name}.");

    private static void ValidateRegionCopyDescriptors(
        string operation,
        BufferViewDescriptor source,
        BufferViewDescriptor destination)
    {
        if (source.Rank != destination.Rank)
        {
            throw new NotSupportedException(
                $"NTT {operation} source/destination rank mismatch: {source.Rank} vs {destination.Rank}.");
        }

        if (source.ElemType != destination.ElemType)
        {
            throw new NotSupportedException(
                $"NTT {operation} source/destination dtype mismatch: {source.ElemType} vs {destination.ElemType}.");
        }
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

    private sealed class ReductionState
    {
        public ReductionState(
            Call call,
            ReductionKernelKind kind,
            BaseExpr[] outputs,
            string[] accumulators,
            string initialized,
            string? elementCount,
            int expectedUpdateCount)
        {
            Call = call;
            Kind = kind;
            Outputs = outputs;
            Accumulators = accumulators;
            Initialized = initialized;
            ElementCount = elementCount;
            ExpectedUpdateCount = expectedUpdateCount;
        }

        public Call Call { get; }

        public ReductionKernelKind Kind { get; }

        public BaseExpr[] Outputs { get; }

        public string[] Accumulators { get; }

        public string Initialized { get; }

        public string? ElementCount { get; }

        public int ExpectedUpdateCount { get; }

        public int UpdateCount { get; set; }
    }

    private sealed record BufferViewDescriptor(
        string Symbol,
        DataType ElemType,
        string[] Shape,
        string[] GlobalOffsets)
    {
        public int Rank => Shape.Length;
    }

    private static void ValidateQKVParallelLinearScales(IReadOnlyList<BaseExpr> args)
    {
        if (args.Count < 16 || args.Skip(7).Take(6).Any(arg => arg is not None))
        {
            throw new NotSupportedException("NTT QKVParallelLinear codegen currently supports only None input/weight scales.");
        }
    }

    private static void ValidateMatMulGluScales(IReadOnlyList<BaseExpr> args)
    {
        if (args.Count < 10 || args.Skip(5).Take(4).Any(arg => arg is not None))
        {
            throw new NotSupportedException("NTT MatMulGlu codegen currently supports only None input/weight scales.");
        }
    }

    protected override CSymbol VisitVar(Var expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var name = IRHelpers.GetIdentityName(expr.Name);
        var index = VisitEntry.Parameters.IndexOf(expr);
        if (index != -1)
        {
            symbol = new CSymbol($"T{index}", name);
        }
        else
        {
            symbol = new(
                expr.CheckedType switch
                {
                    TensorType t => t.DType.ToC(),
                    AnyType => "auto",
                    _ => throw new ArgumentOutOfRangeException(nameof(expr)),
                },
                expr.Name + "_" + expr.GlobalVarIndex.ToString());
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

        var name = IRHelpers.GetIdentityName(expr.Name);
        var index = VisitEntry.Parameters.IndexOf(expr);
        if (index != -1)
        {
            symbol = new CSymbol($"T{index}", name);
        }
        else
        {
            symbol = new(
                expr.CheckedType switch
                {
                    TensorType tensorType => tensorType.DType.ToC(),
                    DistributedType distributedType => distributedType.TensorType.DType.ToC(),
                    _ => throw new NotSupportedException($"Unsupported buffer var type: {expr.CheckedType}"),
                },
                name);
        }

        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitAsDim(AsDim expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var value = Visit(expr.Dim);
        symbol = new("dim_t", value.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitBufferRegion(BufferRegion expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        // FIXME: Use extents instead of stop in BufferRegion.
        throw new NotImplementedException();
#if false
        var buffer = Visit(expr.Buffer);
        var begins = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Start).Name))}";
        var extents = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Stop).Name))}";
        symbol = new(string.Empty, $"{buffer.Name}.view(make_shape({begins}), make_shape({extents}))");
        _exprMemo.Add(expr, symbol);
        return symbol;
#endif
    }
}
