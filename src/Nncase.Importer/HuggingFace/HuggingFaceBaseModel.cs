// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using Tuple = System.Tuple;
using TypeCode = Nncase.Runtime.TypeCode;

namespace Nncase.Importer;

/// <summary>
/// This model architectures base on LlamaForCausalLM.
/// </summary>
public abstract class HuggingFaceModel
{
    public string WeightsDir = string.Empty;
    public Dictionary<string, string> WeightToFileMap = new();
    public Dictionary<string, Tensor> LoadedWeights = new();
    public HashSet<string> LoadedFiles = new();
    private LayerFunctionBuildContext? _layerFunctionBuildContext;

    protected ModelInitContext? Context { get; private set; }

    protected Dictionary<string, object> Config => Context?.Config ?? throw new InvalidOperationException("Model not initialized - Config is null");

    protected Dictionary<string, Tensor> ConstTensors => Context?.ConstTensors ?? throw new InvalidOperationException("Model not initialized - ConstTensors is null");

    protected ImportOptions ImportOptions => Context?.ImportOptions ?? throw new InvalidOperationException("Model not initialized - ImportOptions is null");

    protected CompileSession CompileSession => Context?.CompileSession ?? throw new InvalidOperationException("Model not initialized - CompileSession is null");

    protected virtual bool SupportsDecoderLayerFunctionReuse => true;

    private sealed class LayerFunctionBuildContext
    {
        private readonly Dictionary<string, Expr> _parametersByWeight = new();

        public List<IVar> Parameters { get; } = new();

        public List<LayerParameterSpec> ParameterSpecs { get; } = new();

        public Expr GetOrAddParameter(string weightName, Expr templateValue, Func<int, Expr> argumentFactory)
        {
            if (_parametersByWeight.TryGetValue(weightName, out var parameter))
            {
                return parameter;
            }

            var varName = $"layer_{SanitizeParameterName(GetLayerWeightSuffix(weightName))}";
            var var = new Var(varName, templateValue.CheckedType);
            Parameters.Add(var);
            ParameterSpecs.Add(new LayerParameterSpec(var, argumentFactory));
            _parametersByWeight.Add(weightName, var);
            return var;
        }

        private static string SanitizeParameterName(string name)
        {
            var chars = name.Select(ch => char.IsAsciiLetterOrDigit(ch) ? ch : '_').ToArray();
            return new string(chars).Trim('_');
        }
    }

    private sealed record LayerParameterSpec(IVar Parameter, Func<int, Expr> ArgumentFactory);

    private sealed record DecoderLayerFunction(Function Function, IReadOnlyList<LayerParameterSpec> ParameterSpecs);

    public void Initialize(ModelInitContext context, string dir)
    {
        Context = context ?? throw new ArgumentNullException(nameof(context));
        WeightsDir = dir ?? throw new ArgumentNullException(nameof(dir));
        WeightToFileMap = HuggingFaceUtils.LoadWeightToFileMap(dir);
        LoadedWeights = new Dictionary<string, Tensor>();
        LoadedFiles = new HashSet<string>();
    }

    public Tensor? GetWeight(string weightName)
    {
        if (LoadedWeights.TryGetValue(weightName, out var tensor))
        {
            return tensor;
        }

        if (!WeightToFileMap.TryGetValue(weightName, out var fileName))
        {
            return null;
        }

        if (!LoadedFiles.Contains(fileName))
        {
            var filePath = Path.Combine(WeightsDir, fileName);
            var orgfilePath = filePath.Replace(".safetensors", ".org.safetensors", StringComparison.OrdinalIgnoreCase);
            if (File.Exists(orgfilePath))
            {
                filePath = orgfilePath;
            }

            var tensors = HuggingFaceUtils.LoadAllTensorsFromFile(filePath);
            foreach (var kv in tensors)
            {
                LoadedWeights[kv.Key] = kv.Value;
            }

            LoadedFiles.Add(fileName);
        }

        if (LoadedWeights.TryGetValue(weightName, out tensor))
        {
            return tensor;
        }

        Console.WriteLine($"Weight {weightName} not found after loading {fileName}!");
        throw new InvalidOperationException($"Weight {weightName} could not be loaded from {fileName}");
    }

    private static string GetLayerWeightName(string templateWeightName, int layerIndex)
    {
        const string prefix = "model.layers.";
        if (!templateWeightName.StartsWith(prefix, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Layer function parameter must be a per-layer weight, but got {templateWeightName}.");
        }

        var suffixStart = templateWeightName.IndexOf('.', prefix.Length);
        if (suffixStart < 0)
        {
            throw new InvalidOperationException($"Invalid per-layer weight name: {templateWeightName}.");
        }

        return $"{prefix}{layerIndex}{templateWeightName[suffixStart..]}";
    }

    private static string GetLayerWeightSuffix(string weightName)
    {
        const string prefix = "model.layers.";
        if (!weightName.StartsWith(prefix, StringComparison.Ordinal))
        {
            return weightName;
        }

        var suffixStart = weightName.IndexOf('.', prefix.Length);
        return suffixStart < 0 ? weightName : weightName[(suffixStart + 1)..];
    }

    private Expr GetLayerWeightExpr(string weightName, Func<Tensor, Expr> prepare)
    {
        var tensor = GetWeight(weightName) ?? throw new InvalidOperationException($"Required weight {weightName} is missing.");
        var templateValue = prepare(tensor);
        if (_layerFunctionBuildContext is null)
        {
            return templateValue;
        }

        return _layerFunctionBuildContext.GetOrAddParameter(
            weightName,
            templateValue,
            layerIndex =>
            {
                var layerWeightName = GetLayerWeightName(weightName, layerIndex);
                var layerTensor = GetWeight(layerWeightName) ?? throw new InvalidOperationException($"Required weight {layerWeightName} is missing.");
                return prepare(layerTensor);
            });
    }

    private Expr GetOptionalLayerWeightExpr(string weightName, Func<Tensor, Expr> prepare)
    {
        var tensor = GetWeight(weightName);
        if (tensor is null)
        {
            return None.Default;
        }

        var templateValue = prepare(tensor);
        if (_layerFunctionBuildContext is null)
        {
            return templateValue;
        }

        return _layerFunctionBuildContext.GetOrAddParameter(
            weightName,
            templateValue,
            layerIndex =>
            {
                var layerWeightName = GetLayerWeightName(weightName, layerIndex);
                var layerTensor = GetWeight(layerWeightName);
                return layerTensor is null ? None.Default : prepare(layerTensor);
            });
    }

    public virtual (IEnumerable<IVar> Inputs, Dictionary<IVar, Dimension[]> VarMap) CreateInputs()
    {
        var hiddenSize = (long)Config["hidden_size"];
        _ = (long)Config["num_hidden_layers"];
        var num_attention_heads = (long)Config["num_attention_heads"];
        _ = hiddenSize / num_attention_heads;
        if (Config.ContainsKey("head_dim"))
        {
            _ = (long)Config["head_dim"];
        }

        _ = (long)Config["num_key_value_heads"];

        Context!.Inputs = [];
        Context.DynVarMap = new Dictionary<string, DimVar>();
        var varMap = new Dictionary<IVar, Dimension[]>();

        var bucketOptions = CompileSession.CompileOptions.ShapeBucketOptions;
        Context.FixVarMap = bucketOptions.FixVarMap;

        // local test set
        // _fixVarMap["sequence_length"] = 10;
        // _fixVarMap["history_len"] = 0;
        // TODO: control by config file
        if (!Context.FixVarMap.ContainsKey("sequence_length"))
        {
            Context.DynVarMap["sequence_length"] = new DimVar("sequence_length");
            Context.DynVarMap["sequence_length"].Metadata.Range = new(bucketOptions.RangeInfo["sequence_length"].Min, bucketOptions.RangeInfo["sequence_length"].Max);
        }

        // if (!_fixVarMap.ContainsKey("history_len"))
        // {
        //     _dynVarMap["history_len"] = new DimVar("history_len");
        //     _dynVarMap["history_len"].Metadata.Range=new (4096, 8192);
        // }
        // if (!Context.FixVarMap.ContainsKey("batch_size"))
        // {
        //     Context.DynVarMap["batch_size"] = new DimVar("batch_size");
        //     Context.DynVarMap["batch_size"].Metadata.Range = new(1, 4);
        // }
        var inputIdsShapeExpr = new Dimension[]
        {
            // Context.FixVarMap.ContainsKey("batch_size") ? Context.FixVarMap["batch_size"] : Context.DynVarMap["batch_size"],
            Context.FixVarMap.ContainsKey("sequence_length") ? Context.FixVarMap["sequence_length"] : Context.DynVarMap["sequence_length"],
        };

        // var attentionMaskShapeExpr = new Expr[]
        // {
        //         1L, // _dynVarMap["batch_size"],
        //         20L, // _dynVarMap["sequence_length"]
        // };
        // var positionIdsShapeExpr = new Expr[] {
        //                                         1L, // _dynVarMap["batch_size"],
        //                                         20L, // _dynVarMap["sequence_length"]
        //                                         };

        // // [decode_layers, k_or_v, batch_size, num_heads, past_seq_length, head_dim]
        // var pastKeyValueShapeExpr = new Expr[] { numsHiddenLayers,
        //                                              2L,
        //                                              1L, // _dynVarMap["batch_size"],
        //                                              numKVHeads,
        //                                              0, // _dynVarMap["history_len"],
        //                                              headDim, };
        var inputIds = new Var(
            "input_ids",
            new TensorType(DataTypes.Int64, new RankedShape(inputIdsShapeExpr)));

        // var attentionMask = new Var(
        //     "attention_mask",
        //     new TensorType(
        //         DataTypes.Float32,
        //         new RankedShape(
        //             1L, // _dynVarMap["batch_size"],
        //             20L)));
        // var positionIds = new Var(
        //     "position_ids",
        //     new TensorType(DataTypes.Float32, new RankedShape(
        //                                     1L, // _dynVarMap["batch_size"],
        //                                     20L)));

        // // [decode_layers, k_or_v, batch_size, num_heads, past_seq_length, head_dim]
        // var pastKeyValue = new Var(
        //     "past_key_values",
        //     new TensorType(DataTypes.Float32, new RankedShape(
        //         numsHiddenLayers,
        //         2L,
        //         1L, // _dynVarMap["batch_size"],
        //         numKVHeads,
        //         0, // _dynVarMap["history_len"],
        //         headDim)));
        var pastKeyValue = new Var("kvCache", TensorType.Scalar(
            new ReferenceType(new PagedAttentionKVCacheType { Config = (IPagedAttentionConfig)ImportOptions.HuggingFaceOptions.Config })));

        Context.Inputs.Add(inputIds);
        Context.Inputs.Add(null); // attentionMask
        Context.Inputs.Add(null); // positionIds
        Context.Inputs.Add(pastKeyValue); // pastKeyValue

        // _inputs.Add(attentionMask);
        // _inputs.Add(positionIds);
        // _inputs.Add(pastKeyValue);
        varMap[inputIds] = inputIdsShapeExpr;
        if (!Context.FixVarMap.ContainsKey("sequence_length"))
        {
            varMap[Context.DynVarMap["sequence_length"]] = [Context.DynVarMap["sequence_length"]];
        }

        // varMap[attentionMask] = attentionMaskShapeExpr;
        // varMap[positionIds] = positionIdsShapeExpr;
        // varMap[pastKeyValue] = pastKeyValueShapeExpr;
        var inputs = new List<IVar> { };

        // for the input is optional
        foreach (var input in Context.Inputs)
        {
            if (input != null)
            {
                inputs.Add(input);
            }
        }

        CompileSession.CompileOptions.ShapeBucketOptions.VarMap = varMap;
        return (inputs, varMap);
    }

    // public abstract Expr CreateOutputs();
    public virtual BaseExpr CreateOutputs()
    {
        // TODO: use self.config.output_attention to judge wether output kvache
        Expr? logits = null;
        Expr? lastHiddenStates = null;
        Expr? hiddenStates = null;

        if (ImportOptions.HuggingFaceOptions.OutputLogits)
        {
            logits = Context!.Outputs["logits"];
        }
        else
        {
            lastHiddenStates = Context!.Outputs["lastHiddenStates"];
        }

        if (ImportOptions.HuggingFaceOptions.OutputHiddenStates)
        {
            hiddenStates = Context!.Outputs["hiddenStates"];
        }

        var output = new List<Expr?> { logits, lastHiddenStates, hiddenStates };
        output.RemoveAll(item => item == null);

        return new IR.Tuple([.. output.Where(x => x != null).Cast<Expr>()]);
    }

    public virtual Expr RepeatKV(Expr hiddenStates, long nRep)
    {
        /*
            batch, num_key_value_heads, slen, head_dim = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
            return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        */
        if (nRep == 1)
        {
            return hiddenStates;
        }

        var batch_size = hiddenStates.CheckedShape[0];
        var numKVHeads = hiddenStates.CheckedShape[1];
        var seqLen = hiddenStates.CheckedShape[2];
        var headDim = hiddenStates.CheckedShape[3];
        hiddenStates = IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 2 });

        var tmp = new RankedShape(batch_size, numKVHeads, nRep, seqLen, headDim);
        hiddenStates = IR.F.Tensors.Expand(hiddenStates, tmp);
        hiddenStates = IR.F.Tensors.Reshape(hiddenStates, new RankedShape(batch_size, numKVHeads * nRep, seqLen, headDim));
        return hiddenStates;
    }

    public virtual System.Tuple<Call, Call> ApplyRotaryPosEmb(Expr q, Expr k, Expr cos, Expr sin, long unSqueezeDim = 1)
    {
        // q_embed = (q * cos) + (rotate_half(q) * sin)
        // k_embed = (k * cos) + (rotate_half(k) * sin)
        cos = IR.F.Tensors.Cast(cos, DataTypes.Float32);
        sin = IR.F.Tensors.Cast(sin, DataTypes.Float32);
        var qEmbed = IR.F.NN.RoPE(q, cos, sin);
        var kEmbed = IR.F.NN.RoPE(k, cos, sin);
        return System.Tuple.Create(qEmbed, kEmbed);
    }

    // def rotate_half(x):
    // """Rotates half the hidden dims of the input."""
    // x1 = x[..., : x.shape[-1] // 2]
    // x2 = x[..., x.shape[-1] // 2 :]
    // return torch.cat((-x2, x1), dim=-1)
    public virtual Call RotateHalf(Expr x)
    {
        var xS3 = x.CheckedShape[^1];
        var x1 = IR.F.Tensors.Slice(
            x,
            new[] { 0L },
            new RankedShape(xS3 / 2L),
            new[] { -1L },
            new[] { 1L });
        var x2 = IR.F.Tensors.Slice(
            x,
            new RankedShape(xS3 / 2L),
            new RankedShape(xS3),
            new[] { -1L },
            new[] { 1L });

        return IR.F.Tensors.Concat(new IR.Tuple(IR.F.Math.Neg(x2), x1), -1);
    }

    public virtual Call LLMLayerNorm(Expr hiddenStates, string layerName)
    {
        var originDtype = hiddenStates.CheckedDataType;
        var weightTensor = GetWeight($"{layerName}")!;
        var weight = GetLayerWeightExpr(layerName, tensor => tensor.CastTo(originDtype));
        var bias = Tensor.Zeros(originDtype, weightTensor.Dimensions);
        int axis = -1;

        float eps = 1e-6F;
        if (Config.ContainsKey("rms_norm_eps"))
        {
            eps = (float)Config.GetNestedValue<double>("rms_norm_eps");
        }

        return IR.F.NN.LayerNorm(axis, eps, hiddenStates, weight, bias, false).With(metadata: new IRMetadata() { OutputNames = new[] { layerName.Substring(0, layerName.Length - 7) } });
    }

    public virtual Call Linear(Expr expr, Tensor weight, Tensor? bias = null, Tensor? scaleIf = null, Tensor? scaleW = null, string layerName = "")
    {
        if (scaleIf is not null && scaleW is not null)
        {
            // TODO: only support by tensor quant now!
            if (scaleIf.Rank > 1 || scaleW.Rank > 1)
            {
                throw new NotImplementedException("only support by tensor quant now: ");
            }

            var qScaleA = 1.0f / scaleIf.ToArray<float>()[0];
            var qScaleB = 1.0f / scaleW.ToArray<float>()[0];
            var deqScaleA = 1.0f / qScaleA;
            var deqScaleB = 1.0f / qScaleB;

            var qInput = expr.CheckedDataType switch
            {
                var t when t == DataTypes.BFloat16 => Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, (BFloat16)qScaleA),
                var t when t == DataTypes.Float16 => Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, (Half)qScaleA),
                _ => Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, qScaleA),
            };

            qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Float8E4M3);
            var qWeights = PrepareLinearWeightTensor(weight, DataTypes.Float8E4M3);
            var result = Nncase.IR.F.Math.MatMul(qInput, qWeights, expr.CheckedDataType, deqScaleA * deqScaleB).With(metadata: new IRMetadata() { OutputNames = new[] { layerName } });

            // var result = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, deqScaleA * deqScaleB);
            // result = Nncase.IR.F.Tensors.Cast(result, expr.CheckedDataType);
            if (bias != null)
            {
                bias = bias.CastTo(expr.CheckedDataType);
                result = IR.F.Math.Add(result, bias);
            }

            return result;
        }
        else if (scaleIf is null && scaleW is not null)
        {
            var exprType = expr.CheckedDataType;
            long[] axes = new long[] { expr.CheckedShape.Rank - 1 };
            var max = Nncase.IR.F.Tensors.ReduceMax(expr, axes, float.MinValue, true);
            var min = Nncase.IR.F.Tensors.ReduceMin(expr, axes, float.MaxValue, true);
            var limit = Nncase.IR.F.Math.Max(Nncase.IR.F.Math.Abs(max), Nncase.IR.F.Math.Abs(min));
            if (limit.CheckedDataType != DataTypes.Float32)
            {
                limit = Nncase.IR.F.Tensors.Cast(limit, DataTypes.Float32);
            }

            var qScaleA = Nncase.IR.F.Math.Div((float)Float8E4M3.MaxNormal, limit);
            var deqScaleA = Nncase.IR.F.Math.Div(1.0f, qScaleA);
            var deqScaleB = scaleW;

            if (qScaleA.CheckedDataType != expr.CheckedDataType)
            {
                qScaleA = Nncase.IR.F.Tensors.Cast(qScaleA, expr.CheckedDataType);
            }

            var qInput = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, qScaleA);
            qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Float8E4M3);
            var qWeights = PrepareLinearWeightTensor(weight, DataTypes.Float8E4M3);
            var qMatmul = Nncase.IR.F.Math.MatMul(qInput, qWeights, DataTypes.Float32).With(metadata: new IRMetadata() { OutputNames = new[] { layerName } });

            if (deqScaleA.CheckedDataType != qMatmul.CheckedDataType)
            {
                deqScaleA = Nncase.IR.F.Tensors.Cast(deqScaleA, qMatmul.CheckedDataType);
            }

            var result = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, deqScaleA);

            if (deqScaleB.Rank == 2)
            {
                long[] dims = System.Linq.Enumerable.Range(0, qMatmul.CheckedShape.Rank).Select(i => 1L).ToArray();
                dims[dims.Length - 1] = deqScaleB.Shape[0].FixedValue;
                deqScaleB = Tensor.From<float>(deqScaleB.ToArray<float>(), dims);
                if (deqScaleB.ElementType != qMatmul.CheckedDataType)
                {
                    deqScaleB = deqScaleB.CastTo(qMatmul.CheckedDataType);
                }
            }

            result = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, result, deqScaleB);
            result = Nncase.IR.F.Tensors.Cast(result, exprType);
            if (bias != null)
            {
                result = IR.F.Math.Add(result, bias);
            }

            return result;
        }
        else
        {
            var transposed_weight = PrepareLinearWeightTensor(weight, expr.CheckedDataType);
            var result = IR.F.Math.MatMul(expr, transposed_weight, expr.CheckedDataType).With(metadata: new IRMetadata() { OutputNames = new[] { layerName } });
            if (bias != null)
            {
                result = IR.F.Math.Add(result, bias);
            }

            return result;
        }
    }

    protected virtual Call LinearByName(
        Expr expr,
        string weightName,
        string? biasName = null,
        string? inputScaleName = null,
        string? weightScaleName = null,
        string layerName = "")
    {
        var inputScaleTensor = inputScaleName is null ? null : GetWeight(inputScaleName);
        var weightScaleTensor = weightScaleName is null ? null : GetWeight(weightScaleName);
        var hasInputScale = inputScaleTensor is not null;
        var hasWeightScale = weightScaleTensor is not null;
        var preparedWeightType = hasWeightScale ? DataTypes.Float8E4M3 : expr.CheckedDataType;
        var weight = GetLayerWeightExpr(weightName, tensor => PrepareLinearWeightTensor(tensor, preparedWeightType));
        var bias = biasName is null
            ? None.Default
            : GetOptionalLayerWeightExpr(biasName, tensor => tensor.CastTo(expr.CheckedDataType));
        var inputScale = hasInputScale
            ? GetLayerWeightExpr(inputScaleName!, tensor => tensor)
            : (Expr)None.Default;
        var weightScale = hasWeightScale
            ? GetLayerWeightExpr(weightScaleName!, tensor => tensor)
            : (Expr)None.Default;
        return BuildLinear(expr, weight, bias, inputScale, weightScale, hasInputScale, hasWeightScale, layerName);
    }

    private static Call BuildLinear(
        Expr expr,
        Expr weight,
        Expr bias,
        Expr inputScale,
        Expr weightScale,
        bool hasInputScale,
        bool hasWeightScale,
        string layerName)
    {
        if (hasInputScale && hasWeightScale)
        {
            var qScaleA = Nncase.IR.F.Math.Div(1.0f, inputScale);
            if (qScaleA.CheckedDataType != expr.CheckedDataType)
            {
                qScaleA = Nncase.IR.F.Tensors.Cast(qScaleA, expr.CheckedDataType);
            }

            var qInput = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, qScaleA);
            qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Float8E4M3);
            var deqScale = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, inputScale, weightScale);
            var result = Nncase.IR.F.Math.MatMul(qInput, weight, expr.CheckedDataType, deqScale).With(metadata: new IRMetadata() { OutputNames = new[] { layerName } });

            if (bias is not None)
            {
                result = IR.F.Math.Add(result, bias);
            }

            return result;
        }
        else if (!hasInputScale && hasWeightScale)
        {
            var exprType = expr.CheckedDataType;
            long[] axes = new long[] { expr.CheckedShape.Rank - 1 };
            var max = Nncase.IR.F.Tensors.ReduceMax(expr, axes, float.MinValue, true);
            var min = Nncase.IR.F.Tensors.ReduceMin(expr, axes, float.MaxValue, true);
            var limit = Nncase.IR.F.Math.Max(Nncase.IR.F.Math.Abs(max), Nncase.IR.F.Math.Abs(min));
            if (limit.CheckedDataType != DataTypes.Float32)
            {
                limit = Nncase.IR.F.Tensors.Cast(limit, DataTypes.Float32);
            }

            var qScaleA = Nncase.IR.F.Math.Div((float)Float8E4M3.MaxNormal, limit);
            var deqScaleA = Nncase.IR.F.Math.Div(1.0f, qScaleA);
            var deqScaleB = weightScale;

            if (qScaleA.CheckedDataType != expr.CheckedDataType)
            {
                qScaleA = Nncase.IR.F.Tensors.Cast(qScaleA, expr.CheckedDataType);
            }

            var qInput = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, qScaleA);
            qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Float8E4M3);
            var qMatmul = Nncase.IR.F.Math.MatMul(qInput, weight, DataTypes.Float32).With(metadata: new IRMetadata() { OutputNames = new[] { layerName } });

            if (deqScaleA.CheckedDataType != qMatmul.CheckedDataType)
            {
                deqScaleA = Nncase.IR.F.Tensors.Cast(deqScaleA, qMatmul.CheckedDataType);
            }

            var result = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, deqScaleA);

            if (deqScaleB.CheckedShape.Rank == 2)
            {
                var dims = Enumerable.Range(0, qMatmul.CheckedShape.Rank).Select(_ => (Dimension)1L).ToArray();
                dims[^1] = deqScaleB.CheckedShape[0];
                deqScaleB = IR.F.Tensors.Reshape(deqScaleB, new RankedShape(dims));
            }

            if (deqScaleB.CheckedDataType != qMatmul.CheckedDataType)
            {
                deqScaleB = IR.F.Tensors.Cast(deqScaleB, qMatmul.CheckedDataType);
            }

            result = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, result, deqScaleB);
            result = Nncase.IR.F.Tensors.Cast(result, exprType);
            if (bias is not None)
            {
                result = IR.F.Math.Add(result, bias);
            }

            return result;
        }
        else
        {
            var result = IR.F.Math.MatMul(expr, weight, expr.CheckedDataType).With(metadata: new IRMetadata() { OutputNames = new[] { layerName } });
            if (bias is not None)
            {
                result = IR.F.Math.Add(result, bias);
            }

            return result;
        }
    }

    public virtual Tuple<Expr, Expr> DecodeLayer(
            int count,
            Expr hiddenStates,
            Expr pastKeyValues,
            Tuple<Expr, Expr> positionEmbeddings,
            Dimension layerId)
    {
        var residual = hiddenStates;
        hiddenStates = LLMLayerNorm(
            hiddenStates,
            $"model.layers.{count}.input_layernorm.weight");

        // TODO: using `config.attn_implementation` to choose attention implementation
        // self attention
        (hiddenStates, pastKeyValues) = LLMSelfAttention(
            count,
            hiddenStates,
            pastKeyValues,
            positionEmbeddings,
            layerId);
        hiddenStates = residual + hiddenStates;

        // fully Connected
        residual = hiddenStates;
        hiddenStates = LLMLayerNorm(
            hiddenStates,
            $"model.layers.{count}.post_attention_layernorm.weight");

        hiddenStates = LLMMlp(count, hiddenStates);

        hiddenStates = residual + hiddenStates;

        var output = hiddenStates;

        return System.Tuple.Create<Expr, Expr>(output, pastKeyValues);
    }

    public virtual Call LLMMlp(int count, Expr hiddenStates)
    {
        return LinearByName(
            BuildMatMulGlu(count, hiddenStates),
            $"model.layers.{count}.mlp.down_proj.weight",
            inputScaleName: $"model.layers.{count}.mlp.down_proj.input_scale",
            weightScaleName: $"model.layers.{count}.mlp.down_proj.weight_scale",
            layerName: $"model.layers.{count}.mlp.down_proj");
    }

    public virtual Tuple<Call, Call, Call> QKVCompute(int count, Expr hiddenStates, Dimension seqLen, Dimension headDim)
    {
        var hidden_shape = new RankedShape(seqLen, -1L, headDim);
        return BuildQKVParallelLinear(count, hiddenStates, hidden_shape);
    }

    public virtual Tuple<Expr, Expr> EagerAttentionForward(Expr query, Expr key, Expr value, Expr? attentionMask, float scaling)
    {
        var numKVGroups = (long)Config["num_attention_heads"] / (long)Config["num_key_value_heads"];
        var keyStates = RepeatKV(key, numKVGroups);
        var valueStates = RepeatKV(value, numKVGroups);
        var scalingExpr = IR.F.Tensors.Cast(Tensor.FromScalar(scaling), query.CheckedDataType);
        Expr attnWeights = IR.F.Math.MatMul(query, IR.F.Tensors.Transpose(keyStates, ShapeUtility.GetPermutation(keyStates, [2, 3])), query.CheckedDataType).With(metadata: new IRMetadata() { OutputNames = new[] { "EagerAttentionForward0" } }) * scalingExpr;
        if (attentionMask is not null)
        {
            var causalMask = IR.F.Tensors.Slice(
                    attentionMask,
                    new[] { 0L },
                    new RankedShape(keyStates.CheckedShape[^2]),
                    new[] { 3L },
                    new[] { 1L });

            attnWeights += causalMask;
        }

        attnWeights = IR.F.Tensors.Cast(IR.F.NN.Softmax(IR.F.Tensors.Cast(attnWeights, DataTypes.Float32), 3L), valueStates.CheckedDataType);

        Expr attnOutput = IR.F.Math.MatMul(attnWeights, valueStates, query.CheckedDataType).With(metadata: new IRMetadata() { OutputNames = new[] { "EagerAttentionForward1" } });
        attnOutput = IR.F.Tensors.Transpose(attnOutput, ShapeUtility.GetPermutation(attnOutput, [1, 2]));

        // TODO: base on config to decide output attnWeights or not
        return System.Tuple.Create(attnOutput, attnWeights);
    }

    public virtual Tuple<Expr, Expr> RotaryEmbedding(Expr x, Expr kvObject, float[] invFreq, float attentionScaling)
    {
        var positionIds = IR.F.NN.GetPositionIds(x.CheckedShape[0], kvObject);
        positionIds = IR.F.Tensors.Unsqueeze(IR.F.Tensors.Cast(positionIds, DataTypes.Float32), [1]);
        var invFreqExpanded = invFreq.Concat(invFreq).ToArray();
        var emb = IR.F.Math.Mul(invFreqExpanded, positionIds).With(metadata: new IRMetadata()
        {
            OutputNames = ["RotaryEmbedding"],
        });

        // add attention scaling
        Expr cos = IR.F.Math.Unary(UnaryOp.Cos, emb) * attentionScaling;
        Expr sin = IR.F.Math.Unary(UnaryOp.Sin, emb) * attentionScaling;

        cos = IR.F.Tensors.Unsqueeze(cos, [1]);
        sin = IR.F.Tensors.Unsqueeze(sin, [1]);

        return System.Tuple.Create(cos, sin);
    }

    public virtual Tuple<Call, Call> UpdateKVWithCache(int layerIdx, Call k, Call v, Expr pastKeyValues)
    {
        // dynamic cache update kvcache
        /*
            # Update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            # Update the cache
            if key_states is not None:
                if len(self.key_cache) <= layer_idx:
                    # There may be skipped layers, fill them with empty lists
                    for _ in range(len(self.key_cache), layer_idx):
                        self.key_cache.append([])
                        self.value_cache.append([])
                    self.key_cache.append(key_states)
                    self.value_cache.append(value_states)
                elif (
                    len(self.key_cache[layer_idx]) == 0
                ):  # fills previously skipped layers; checking for tensor causes errors
                    self.key_cache[layer_idx] = key_states
                    self.value_cache[layer_idx] = value_states
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        */
        // past_key_values shape: [decode_layers, k_or_v, batch_size, num_heads, past_seq_length, head_dim]
        var pastKeyValuesCurrentLayer = IR.F.Tensors.Gather(pastKeyValues, 0, (long)layerIdx);
        var pastKeyCurrentLayer = IR.F.Tensors.Gather(pastKeyValuesCurrentLayer, 0, 0L);
        var pastValueCurrentLayer = IR.F.Tensors.Gather(pastKeyValuesCurrentLayer, 0, 1L);

        // [batch_size, num_heads, past_seq_length, head_dim]
        var key_states = IR.F.Tensors.Concat(new IR.Tuple(pastKeyCurrentLayer, k), -2);
        var value_states = IR.F.Tensors.Concat(new IR.Tuple(pastValueCurrentLayer, v), -2);

        return System.Tuple.Create(key_states, value_states);
    }

    public virtual Expr MergeKV(Expr key, Expr value)
    {
        // [batchsize, num_heads, seq_length, head_dim]  ->[1,2,batchsize, num_heads, seq_length, head_dim]
        var keyStates = IR.F.Tensors.Unsqueeze(key, new long[] { 0 });
        var valueStates = IR.F.Tensors.Unsqueeze(value, new long[] { 0 });
        var mergedKeyValue = IR.F.Tensors.Concat(new IR.Tuple(keyStates, valueStates), 0);
        return IR.F.Tensors.Unsqueeze(mergedKeyValue, new long[] { 0 });
    }

    public virtual Expr Prepare4dCausalAttentionMaskWithCachePosition(
                            Expr? attentionMask,
                            Dimension seqLen,
                            Dimension targtLen,
                            DataType dtype,
                            Expr cachePosition,
                            Dimension batchSize,
                            Expr? pastKeyValues)
    {
        Expr? casualMask;
        if (attentionMask != null && attentionMask.CheckedShape.Rank == 4)
        {
            Console.WriteLine("attention_mask is already 4D, no need to prepare 4D causal mask.");
            casualMask = attentionMask;
        }
        else
        {
            var mask_shape = new RankedShape([seqLen, targtLen]);
            Tensor minValue;

            // get the min value for current dtype
            FieldInfo? minValueField = dtype.CLRType.GetField("MinValue", BindingFlags.Public | BindingFlags.Static);
            if (minValueField != null)
            {
                var min = minValueField.GetValue(null)!;
                minValue = Tensor.FromScalar(dtype, min, [1L]);
            }
            else
            {
                PropertyInfo? minValueProperty = dtype.CLRType.GetProperty("MinValue", BindingFlags.Public | BindingFlags.Static);
                if (minValueProperty != null)
                {
                    var min = minValueProperty.GetValue(null)!;
                    minValue = Tensor.FromScalar(dtype, min, [1L]);
                }
                else
                {
                    throw new InvalidOperationException($"cannot get current dtype's min value:{dtype.CLRType}");
                }
            }

            casualMask = IR.F.Tensors.ConstantOfShape(mask_shape, minValue);

            /*
                min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            ;
            */
            var diagonalAttendMask = IR.F.Tensors.Range(0L, IR.F.Shapes.AsTensor(targtLen), 1L) > IR.F.Tensors.Reshape(cachePosition, new long[] { -1, 1 });

            // TODO: maybe consider:
            /*
             if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            */

            // casualMask = casualMask * IR.F.Tensors.Cast(diagonalAttendMask, casualMask.CheckedDataType);
            casualMask = IR.F.Tensors.Where(diagonalAttendMask, casualMask, IR.F.Tensors.Cast(0f, casualMask.CheckedDataType));

            // casualMask = casualMask[None, None, :, :].expand(batch_size, 1, -1, -1)
            var expandShape = new RankedShape(batchSize, 1L, seqLen, targtLen);
            casualMask = IR.F.Tensors.Unsqueeze(casualMask, new long[] { 0, 1 });
            casualMask = IR.F.Tensors.Expand(casualMask, expandShape);
            /*
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
            */
            if (attentionMask != null)
            {
                var maskLength = attentionMask.CheckedShape[^1];
                var paddingMask = IR.F.Tensors.Slice(
                    casualMask,
                    new[] { 0L, 0L, 0L, 0L },
                    new RankedShape(maskLength),
                    new[] { 0L, 1L, 2L, 3L },
                    new[] { 1L, 1L, 1L, 1L });
                paddingMask += IR.F.Tensors.Unsqueeze(attentionMask, new long[] { 1, 2 });

                /*
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )
                */
                paddingMask = IR.F.Math.Equal(paddingMask, 0.0f);
                var maskPart = IR.F.Tensors.Slice(
                    casualMask,
                    new[] { 0L },
                    new RankedShape(maskLength),
                    new[] { -1L },
                    new[] { 1L });

                var minDtypeMatrix = IR.F.Tensors.ConstantOfShape(maskPart.CheckedShape, minValue);

                maskPart = IR.F.Tensors.Where(paddingMask, minDtypeMatrix, maskPart);

                // TODO: for dynamic cache, maskLength== sequence length == target length
                //  just return maskPart
                var leftPart = IR.F.Tensors.Slice(
                    casualMask,
                    new RankedShape(maskLength),
                    new RankedShape(casualMask.CheckedShape[^1]),
                    new[] { -1L },
                    new[] { 1L });
                casualMask = IR.F.Tensors.Concat(new IR.Tuple(maskPart, leftPart), -1);
            }
        }

        return casualMask;
    }

    public virtual Expr UpdatecasualMask(Expr? attentionMask, Expr inputsEmbeds, Expr cachePosition, Expr? pastKeyValues, bool outputAttentions = false)
    {
        /*
        # SlidingWindowCache or StaticCache
    if using_sliding_window_cache or using_static_cache:
        target_length = past_key_values.get_max_cache_shape()
    # DynamicCache or no cache
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )
        */
        // TODO:consider flash attention v2
        Dimension historyLen = 0L;

        // if (pastKeyValues != null)
        // {
        //     // FIXME: use api to get historyLen.
        //     historyLen = pastKeyValues.CheckedShape[-2];
        // }
        var batchSize = inputsEmbeds.CheckedShape[0];
        var seqLen = inputsEmbeds.CheckedShape[1];
        var targetLength = historyLen + seqLen + 1L;
        if (attentionMask != null)
        {
            targetLength = attentionMask.CheckedShape[^1];
        }

        var dtype = inputsEmbeds.CheckedDataType;

        Expr casualMask = Prepare4dCausalAttentionMaskWithCachePosition(
                                            attentionMask,
                                            seqLen,
                                            targetLength,
                                            dtype,
                                            cachePosition,
                                            batchSize,
                                            pastKeyValues);
        /*
        if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type in ["cuda", "xpu"]
        and not output_attentions
        ):
        # Attend to all tokens in fully masked rows in the casualMask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        casualMask = AttentionMaskConverter._unmask_unattended(casualMask, min_dtype)
        */

        // TODO: maybe need upon
        return casualMask;
    }

    public virtual Call Embedding(Expr input, Tensor embedingWeight, long? paddingIdx = null)
    {
        var gatherResult = IR.F.Tensors.Gather(embedingWeight, 0, input);
        if (paddingIdx == null)
        {
            return gatherResult;
        }
        else
        {
            var zeros = Tensor.Zeros(embedingWeight.ElementType, [1]);
            var paddingMask = IR.F.Math.Equal(input, paddingIdx);
            paddingMask = IR.F.Tensors.Unsqueeze(paddingMask, [1]);
            var results = IR.F.Tensors.Where(paddingMask, zeros, gatherResult);
            return results;
        }
    }

    public virtual Tuple<Expr, Expr> LLMSelfAttention(
                int count,
                Expr hiddenStates,
                Expr paskKeyValues,
                Tuple<Expr, Expr> positionEmbeddings,
                Dimension layerId)
    {
        var head_dim = (long)Context!.Config!["hidden_size"] / (long)Context.Config["num_attention_heads"];
        if (Context.Config!.Keys.Contains("head_dim"))
        {
            head_dim = (long)Context.Config["head_dim"];
        }

        var pagedAttentionConfig = (IPagedAttentionConfig)Context.ImportOptions!.HuggingFaceOptions.Config;

        // var batch_size = hiddenStates.CheckedShape[0];
        var seq_len = hiddenStates.CheckedShape[0];
        var (queryStates, keyStates, valueStates) = QKVCompute(count, hiddenStates, seq_len, head_dim);

        var (cos, sin) = positionEmbeddings;

        // // apply_rotary_pos_emb
        (queryStates, keyStates) = ApplyRotaryPosEmb(queryStates, keyStates, cos, sin);
        var qType = ((Expr)queryStates).CheckedDataType;
        AttentionDimKind[] qSrcLayout = [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim];
        AttentionDimKind[] kvSrcLayout = [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim];
        {
            var kvDestLayout = GetPagedAttentionTensorLayout(pagedAttentionConfig, kvSrcLayout);
            var kvPerms = ModelUtils.GetLayoutPerm(kvSrcLayout, kvDestLayout);
            var (keyLanes, keyVectorizedAxis) = ModelUtils.GetQKVVectorizeParams(pagedAttentionConfig, kvDestLayout, AttentionCacheKind.Key);
            var (valueLanes, valueVectorizedAxis) = ModelUtils.GetQKVVectorizeParams(pagedAttentionConfig, kvDestLayout, AttentionCacheKind.Value);
            var transK = TransposeIfNeeded(keyStates, kvPerms);
            var castK = pagedAttentionConfig.KVPrimType != qType ? IR.F.Tensors.Cast(transK, pagedAttentionConfig.KVPrimType) : transK;
            var vectorizedK = keyLanes.Length > 0 ? IR.F.Tensors.Pack(castK, keyLanes, keyVectorizedAxis) : castK;
            paskKeyValues = IR.F.NN.UpdatePagedAttentionKVCache(vectorizedK, paskKeyValues, layerId, AttentionCacheKind.Key, kvDestLayout);

            var transV = TransposeIfNeeded(valueStates, kvPerms);
            var castV = pagedAttentionConfig.KVPrimType != qType ? IR.F.Tensors.Cast(transV, pagedAttentionConfig.KVPrimType) : transV;
            var vectorizedV = valueLanes.Length > 0 ? IR.F.Tensors.Pack(castV, valueLanes, valueVectorizedAxis) : castV;
            paskKeyValues = IR.F.NN.UpdatePagedAttentionKVCache(vectorizedV, paskKeyValues, layerId, AttentionCacheKind.Value, kvDestLayout);
        }

        var scaling = Tensor.FromScalar((float)(1.0f / System.Math.Sqrt((double)head_dim)));

        // var mergedKeyValue = MergeKV(keyStates, valueStates);
        var qDestLayout = GetPagedAttentionTensorLayout(pagedAttentionConfig, qSrcLayout);
        var qPerm = ModelUtils.GetLayoutPerm(qSrcLayout, qDestLayout);
        var (qLanes, qVectorizedAxis) = ModelUtils.GetQKVVectorizeParams(pagedAttentionConfig, qDestLayout);
        bool isXpu = Context.CompileSession!.Target.Name == "xpu";

        // if (isXpu)
        // {
        //     var padding_m = Dimension.AlignUp(seq_len, 8) - seq_len;
        //     queryStates = seq_len is DimVar ? IR.F.NN.Pad(queryStates, new(new(0, 0), new(0, padding_m), new(0, 0)), PadMode.Constant, Tensor.Zero(queryStates.CheckedDataType)) : queryStates;
        // }
        var transQ = TransposeIfNeeded(queryStates, qPerm);
        var castQ = pagedAttentionConfig.KVPrimType != qType ? IR.F.Tensors.Cast(transQ, pagedAttentionConfig.KVPrimType) : transQ;
        var vectorizedQ = qLanes.Length > 0 ? IR.F.Tensors.Pack(castQ, qLanes, qVectorizedAxis) : castQ;

        // cpu : [q_head, max_query_len, max_seq_len + 1 ]<primtype>
        var extra_size = pagedAttentionConfig.KVPrimType.SizeInBytes * (long)Context.Config["num_attention_heads"] * Context.ImportOptions.HuggingFaceOptions.MaxModelLen * (Context.ImportOptions.HuggingFaceOptions.MaxModelLen + 1);

        // xpu : 10 mb.
        if (isXpu)
        {
            extra_size = 10 * 1024 * 1024;
        }

        var hidden_size = Context!.Config.ContainsKey("head_dim") ? (int)((long)Context!.Config!["num_attention_heads"] * (long)Context!.Config!["head_dim"]) : (int)(long)Context!.Config!["hidden_size"];
        var output = IR.F.NN.PagedAttention(
            vectorizedQ,
            paskKeyValues,
            IR.F.Buffer.Uninitialized(DataTypes.UInt8, TIR.MemoryLocation.Data, [extra_size]),
            scaling.CastTo(pagedAttentionConfig.KVPrimType, CastMode.KDefault),
            layerId,
            qDestLayout,
            hidden_size);

        output = qLanes.Length > 0 ? IR.F.Tensors.Unpack(output, qLanes, qVectorizedAxis) : output;

        output = pagedAttentionConfig.KVPrimType != qType ? IR.F.Tensors.Cast(output, qType) : output;
        output = TransposeIfNeeded(output, ModelUtils.GetLayoutPerm(qDestLayout, qSrcLayout));

        // if (isXpu)
        // {
        //     output = seq_len is DimVar ? IR.F.Tensors.Slice(output, new[] { 0 }, new Dimension[] { seq_len }, new[] { 1 }, new[] { 1 }) : output;
        // }
        output = IR.F.Tensors.Reshape(output, new RankedShape(seq_len, -1L));
        output = LinearByName(
            output,
            $"model.layers.{count}.self_attn.o_proj.weight",
            inputScaleName: $"model.layers.{count}.self_attn.o_proj.input_scale",
            weightScaleName: $"model.layers.{count}.self_attn.o_proj.weight_scale",
            layerName: $"model.layers.{count}.self_attn.o_proj");
        return System.Tuple.Create(output, paskKeyValues);
    }

    private DecoderLayerFunction BuildDecoderLayerFunction(
        Expr hiddenStates,
        Expr pastKeyValues,
        Tuple<Expr, Expr> positionEmbeddings)
    {
        var hiddenStatesParam = new Var("hidden_states", hiddenStates.CheckedType);
        var pastKeyValuesParam = new Var("kv_cache", pastKeyValues.CheckedType);
        var layerIdParam = new DimVar("layer_id");
        var cosParam = new Var("cos", positionEmbeddings.Item1.CheckedType);
        var sinParam = new Var("sin", positionEmbeddings.Item2.CheckedType);
        var context = new LayerFunctionBuildContext();
        if (_layerFunctionBuildContext is not null)
        {
            throw new InvalidOperationException("Nested HuggingFace decoder layer function construction is not supported.");
        }

        _layerFunctionBuildContext = context;
        try
        {
            var (layerOutput, updatedKvCache) = DecodeLayer(
                0,
                hiddenStatesParam,
                pastKeyValuesParam,
                System.Tuple.Create<Expr, Expr>(cosParam, sinParam),
                layerIdParam);
            var body = new IR.Tuple(layerOutput, updatedKvCache);
            var parameters = new List<IVar> { hiddenStatesParam, pastKeyValuesParam, layerIdParam, cosParam, sinParam };
            parameters.AddRange(context.Parameters);
            var function = new Function("hf_decoder_layer", CompileSession.Target.Name, body, parameters.ToArray());
            return new DecoderLayerFunction(function, context.ParameterSpecs.ToArray());
        }
        finally
        {
            _layerFunctionBuildContext = null;
        }
    }

    private static Call CallDecoderLayerFunction(
        DecoderLayerFunction layerFunction,
        int layerIndex,
        Expr hiddenStates,
        Expr pastKeyValues,
        Tuple<Expr, Expr> positionEmbeddings)
    {
        var arguments = new List<BaseExpr>
        {
            hiddenStates,
            pastKeyValues,
            new DimConst(layerIndex),
            positionEmbeddings.Item1,
            positionEmbeddings.Item2,
        };
        arguments.AddRange(layerFunction.ParameterSpecs.Select(spec => (BaseExpr)spec.ArgumentFactory(layerIndex)));
        return new Call(layerFunction.Function, arguments.ToArray());
    }

    public virtual Tuple<Expr, Expr?> LLMModel(
            Expr inputIds,
            Expr pastKeyValues)
    {
        /*
         * 1.1 embedding
         * self.padding_idx = config.pad_token_id
         * self.vocab_size = config.vocab_size
         * self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
         */
        var embedTokensWeight = GetWeight("model.embed_tokens.weight")!;
        var requestedTensorType = GetRequestedTensorType();
        if (requestedTensorType is not null)
        {
            embedTokensWeight = embedTokensWeight.CastTo(requestedTensorType);
        }

        Expr? inputEmbeds;
        if (inputIds.CheckedShape.Rank > 2 && inputIds.CheckedDataType.IsFloat())
        {
            System.Console.WriteLine("inputIds rank >2 && dtype.isFloat()==true ,regard input_id as embedding...");
            inputEmbeds = inputIds;
        }
        else
        {
            long? padding_idx = null;
            if (Config!.Keys.Contains("pad_token_id"))
            {
                padding_idx = (long)Config["pad_token_id"];
            }

            inputEmbeds = Embedding(inputIds, embedTokensWeight, padding_idx);
        }

        // The embedding output dtype follows the prepared embedding weight dtype.
        Expr hiddenStates = inputEmbeds;

        var (invFreq, attentionScaling) = ModelUtils.RoPEInit(Context!.Config!);
        var positionEmbeddings = RotaryEmbedding(hiddenStates, pastKeyValues, invFreq, attentionScaling);

        Expr? allHiddenStates = null;
        var layerFunction = SupportsDecoderLayerFunctionReuse
            ? BuildDecoderLayerFunction(hiddenStates, pastKeyValues, positionEmbeddings)
            : null;

        for (int i = 0; i < (int)(long)Context!.Config!["num_hidden_layers"]; i++)
        {
            if (Context.ImportOptions!.HuggingFaceOptions.OutputHiddenStates)
            {
                // allHiddenStates.Add(IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 0 }));
                if (i == 0)
                {
                    allHiddenStates = IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 0 });
                }
                else
                {
                    allHiddenStates = IR.F.Tensors.Concat(new IR.Tuple(allHiddenStates!, IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 0 })), 0);
                }
            }

            // var (hiddenStatesTmp, selfAttenWeights) = DecodeLayer(i, hiddenStates, casualMask, positionIds,
            //     pastKeyValues, outputAttentions,
            //     useCache, cachePosition, positionEmbeddings);
            if (layerFunction is not null)
            {
                var layerResult = CallDecoderLayerFunction(
                    layerFunction,
                    i,
                    hiddenStates,
                    pastKeyValues,
                    positionEmbeddings);
                hiddenStates = IR.F.Tensors.GetItem(layerResult, 0);
                pastKeyValues = IR.F.Tensors.GetItem(layerResult, 1);
            }
            else
            {
                var (hiddenStatesTmp, pastKeyValuesTmp) = DecodeLayer(
                    i,
                    hiddenStates,
                    pastKeyValues,
                    positionEmbeddings,
                    new DimConst(i));
                pastKeyValues = pastKeyValuesTmp;
                hiddenStates = hiddenStatesTmp;
            }
        }

        // the last one
        Expr lastHiddenStates = LLMLayerNorm(hiddenStates, "model.norm.weight");

        if (ImportOptions.HuggingFaceOptions.OutputHiddenStates)
        {
            allHiddenStates = IR.F.Tensors.Concat(new IR.Tuple(allHiddenStates!, IR.F.Tensors.Unsqueeze(lastHiddenStates, new long[] { 0 })), 0);
        }

        return Tuple.Create(lastHiddenStates, allHiddenStates);

        // return Tuple.Create(lastHiddenStates, allSelfAttns, allKVcaches);
    }

    public virtual void VisitForCausalLM()
    {
        if (Context!.ConstTensors == null)
        {
            throw new ArgumentNullException(nameof(Context.ConstTensors));
        }

        Var input_ids = Context.Inputs![0]!;
        _ = Context.Inputs[1];
        _ = Context.Inputs[2];
        var pastKeyValues = Context.Inputs![3];

        var (lastHiddenStates, allHiddenStates) = LLMModel(
            input_ids,
            pastKeyValues!);

        var lmHeadWeights = GetWeight("model.embed_tokens.weight")!;
        if (Context!.Config!.ContainsKey("tie_word_embeddings") && !Context!.Config!.GetNestedValue<bool>("tie_word_embeddings"))
        {
            var newLmHeadWeights = GetWeight("lm_head.weight");
            if (newLmHeadWeights != null)
            {
                lmHeadWeights = newLmHeadWeights;
            }
        }

        var lmHead = Linear(lastHiddenStates, lmHeadWeights, layerName: "lm_head");

        // FIXIT: this is work around for bfloat16
        if (Context.ImportOptions!.HuggingFaceOptions.OutputLogits)
        {
            Context.Outputs!.Add("logits", IR.F.Tensors.Cast(lmHead, DataTypes.Float32));
        }
        else
        {
            Context.Outputs!.Add("lastHiddenStates", IR.F.Tensors.Cast(lastHiddenStates, DataTypes.Float32));
        }

        if (Context.ImportOptions.HuggingFaceOptions.OutputHiddenStates)
        {
            // FIXIT: this is work around for bfloat16
            Context.Outputs!["hiddenStates"] = IR.F.Tensors.Cast(allHiddenStates!, DataTypes.Float32);
        }
    }

    protected static Tensor PrepareLinearWeightTensor(Tensor weight, DataType targetType)
    {
        return weight.Transpose([1, 0]).CastTo(targetType);
    }

    protected DataType? GetRequestedTensorType()
    {
        var tensorType = ImportOptions.HuggingFaceOptions.TensorType;
        return string.IsNullOrWhiteSpace(tensorType) || tensorType == "default"
            ? null
            : HuggingFaceUtils.Str2Dtype(tensorType);
    }

    protected virtual Tuple<Call, Call, Call> BuildQKVParallelLinear(int count, Expr hiddenStates, RankedShape hiddenShape)
    {
        string qWeightName = $"model.layers.{count}.self_attn.q_proj.weight";
        string kWeightName = $"model.layers.{count}.self_attn.k_proj.weight";
        string vWeightName = $"model.layers.{count}.self_attn.v_proj.weight";
        string qBiasName = $"model.layers.{count}.self_attn.q_proj.bias";
        string kBiasName = $"model.layers.{count}.self_attn.k_proj.bias";
        string vBiasName = $"model.layers.{count}.self_attn.v_proj.bias";
        string qInputScaleName = $"model.layers.{count}.self_attn.q_proj.input_scale";
        string kInputScaleName = $"model.layers.{count}.self_attn.k_proj.input_scale";
        string vInputScaleName = $"model.layers.{count}.self_attn.v_proj.input_scale";
        string qWeightScaleName = $"model.layers.{count}.self_attn.q_proj.weight_scale";
        string kWeightScaleName = $"model.layers.{count}.self_attn.k_proj.weight_scale";
        string vWeightScaleName = $"model.layers.{count}.self_attn.v_proj.weight_scale";

        Expr PrepareWeight(string weightName, string weightScaleName)
        {
            var weightScale = GetWeight(weightScaleName);
            return GetLayerWeightExpr(weightName, weight => PrepareLinearWeightTensor(weight, weightScale is null ? hiddenStates.CheckedDataType : DataTypes.Float8E4M3));
        }

        Expr PrepareBias(string biasName) => GetOptionalLayerWeightExpr(biasName, bias => bias.CastTo(hiddenStates.CheckedDataType));

        Expr PrepareScale(string scaleName) => GetOptionalLayerWeightExpr(scaleName, scale => scale);

        var numHeads = (long)Config["num_attention_heads"];
        var numKvHeads = (long)Config["num_key_value_heads"];
        var qkvStates = IR.F.NN.QKVParallelLinear(
            hiddenStates,
            PrepareWeight(qWeightName, qWeightScaleName),
            PrepareWeight(kWeightName, kWeightScaleName),
            PrepareWeight(vWeightName, vWeightScaleName),
            PrepareBias(qBiasName),
            PrepareBias(kBiasName),
            PrepareBias(vBiasName),
            PrepareScale(qInputScaleName),
            PrepareScale(kInputScaleName),
            PrepareScale(vInputScaleName),
            PrepareScale(qWeightScaleName),
            PrepareScale(kWeightScaleName),
            PrepareScale(vWeightScaleName),
            numHeads,
            numKvHeads,
            hiddenStates.CheckedDataType)
            .With(metadata: new IRMetadata() { OutputNames = new[] { $"model.layers.{count}.self_attn.qkv_proj" } });
        var queryStates = IR.F.Tensors.Reshape(IR.F.Tensors.GetItem(qkvStates, 0), hiddenShape);
        var keyStates = IR.F.Tensors.Reshape(IR.F.Tensors.GetItem(qkvStates, 1), hiddenShape);
        var valueStates = IR.F.Tensors.Reshape(IR.F.Tensors.GetItem(qkvStates, 2), hiddenShape);
        return System.Tuple.Create(queryStates, keyStates, valueStates);
    }

    protected virtual Call BuildMatMulGlu(int count, Expr hiddenStates)
    {
        string gateWeightName = $"model.layers.{count}.mlp.gate_proj.weight";
        string upWeightName = $"model.layers.{count}.mlp.up_proj.weight";
        string gateBiasName = $"model.layers.{count}.mlp.gate_proj.bias";
        string upBiasName = $"model.layers.{count}.mlp.up_proj.bias";
        string gateInputScaleName = $"model.layers.{count}.mlp.gate_proj.input_scale";
        string upInputScaleName = $"model.layers.{count}.mlp.up_proj.input_scale";
        string gateWeightScaleName = $"model.layers.{count}.mlp.gate_proj.weight_scale";
        string upWeightScaleName = $"model.layers.{count}.mlp.up_proj.weight_scale";

        Expr PrepareWeight(string weightName, string weightScaleName)
        {
            var weightScale = GetWeight(weightScaleName);
            return GetLayerWeightExpr(weightName, weight => PrepareLinearWeightTensor(weight, weightScale is null ? hiddenStates.CheckedDataType : DataTypes.Float8E4M3));
        }

        Expr PrepareBias(string biasName) => GetOptionalLayerWeightExpr(biasName, bias => bias.CastTo(hiddenStates.CheckedDataType));

        Expr PrepareScale(string scaleName) => GetOptionalLayerWeightExpr(scaleName, scale => scale);

        return IR.F.NN.MatMulGlu(
            hiddenStates,
            PrepareWeight(gateWeightName, gateWeightScaleName),
            PrepareWeight(upWeightName, upWeightScaleName),
            PrepareBias(gateBiasName),
            PrepareBias(upBiasName),
            PrepareScale(gateInputScaleName),
            PrepareScale(upInputScaleName),
            PrepareScale(gateWeightScaleName),
            PrepareScale(upWeightScaleName),
            GetMlpGluType(),
            hiddenStates.CheckedDataType)
            .With(metadata: new IRMetadata() { OutputNames = new[] { $"model.layers.{count}.mlp.gate_up_proj" } });
    }

    protected virtual GluType GetMlpGluType()
    {
        var actType = Config.ContainsKey("hidden_act") ? Config.GetNestedValue<string>("hidden_act") : "silu";
        return actType.ToUpperInvariant() switch
        {
            "SILU" or "SWISH" => GluType.SwiGLU,
            _ => throw new NotSupportedException($"MatMulGlu currently supports only SwiGLU, got hidden_act={actType}."),
        };
    }

    /*
    /// <summary>
    /// LLM MoE (Mixture of Experts) layer (base on Qwen3 moe).
    /// </summary>
    /// <param name="count"> layer idx.</param>
    /// <param name="hiddenStates"> query hidden states.</param>
    /// <returns></returns>
    */

    // public virtual Dictionary<string, Call> LLMMoe(int count, Expr hiddenStates)
    // {
    //     var (seqLen, hiddenDim) = (hiddenStates.CheckedShape[0], hiddenStates.CheckedShape[1]);
    //     var expertNum = Config.GetNestedValue<long>("num_experts");
    //     var topK = Config.GetNestedValue<long>("top_k");
    //     var normTopkProb = Config.GetNestedValue<bool>("norm_topk_prob");
    //     // hiddenStates = IR.F.Tensors.Reshape(hiddenStates, new RankedShape(-1, hiddenDim));
    //     var routerW = GetWeight($"model.layers.{count}.mlp.gate.weight")!;
    //     var routerB = GetWeight($"model.layers.{count}.mlp.gate.bias");
    //     var ifScaleRouter = GetWeight($"model.layers.{count}.mlp.gate.input_scale");
    //     var wScaleRouter = GetWeight($"model.layers.{count}.mlp.gate.weight_scale");
    //     var routerLogits = Linear(hiddenStates, routerW, routerB, ifScaleRouter, wScaleRouter, $"model.layers.{count}.mlp.gate");
    //     // [seq_len, expert_num]
    //     var routerLogitsCast = IR.F.Tensors.Cast(routerLogits, DataTypes.Float32);
    //     var routerWeights = IR.F.NN.Softmax(routerLogitsCast, 0);
    //     var topKRes = IR.F.Tensors.TopK(routerWeights, Tensor.FromScalar(topK), -1, true, false);
    //     var (topkRouterWeights, selectedExpert) = (topKRes[0], topKRes[1]);
    //     if (normTopkProb)
    //     {
    //         var denom = IR.F.Tensors.ReduceSum(topkRouterWeights, new[] { -1 }, Tensor.FromScalar(0), keepDims: true);
    //         topkRouterWeights = topkRouterWeights / denom;
    //     }
    //     topkRouterWeights = IR.F.Tensors.Cast(topkRouterWeights, hiddenStates.CheckedDataType);
    //     var finalHiddenStates_ = Tensor.Zeros(hiddenStates.CheckedDataType, [1]);
    //     var finalHiddenStates = IR.F.Tensors.Broadcast(finalHiddenStates_, new RankedShape(seqLen, hiddenDim));
    //     // one hot mask.
    //     // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    //     var expertMask = IR.F.NN.OneHot(
    //         OneHotMode.Normal,
    //         selectedExpert,
    //         expertNum,
    //         Tensor.FromArray(new[] { 0, 1 }),
    //         -1);
    //     expertMask = IR.F.Tensors.Transpose(expertMask, new long[] { 2, 1, 0 });
    //     expertMask = IR.F.Tensors.Cast(expertMask, DataTypes.Float32);
    //     for (int i = 0; i < expertNum; i++)
    //     {
    //         // 获取当前专家的选择概率
    //         var expertMaskSlice = IR.F.Tensors.Slice(expertMask, new[] { (long)i }, new[] { (long)i + 1L }, new[] { 0L }, new[] { 1L }); // [1, topk, seq_len]
    //         expertMaskSlice = IR.F.Tensors.Reshape(expertMaskSlice, new RankedShape(seqLen, topK));
    //         var expertWeights = IR.F.Tensors.ReduceSum(
    //             IR.F.Math.Mul(topkRouterWeights, expertMaskSlice),
    //             new[] { 1L },
    //             Tensor.FromScalar(0.0f),
    //             keepDims: true); // [seq_len, 1]
    //         var shouldCompute = expertWeights > 0.0f;
    //         var maskedHiddenStates = IR.F.Tensors.Where(
    //             shouldCompute,
    //             hiddenStates,
    //             IR.F.Tensors.Cast(0.0f, hiddenStates.CheckedDataType));
    //         ModelUtils.CheckShape(maskedHiddenStates);
    //         var expertOutput = LLMMlp(count, maskedHiddenStates, $"experts.{i}.");
    //         var weightedOutput = IR.F.Math.Mul(expertOutput, expertWeights);
    //         finalHiddenStates = IR.F.Math.Binary(BinaryOp.Add, finalHiddenStates, weightedOutput);
    //     }
    //     finalHiddenStates = IR.F.Tensors.Reshape(finalHiddenStates, new RankedShape(seqLen, hiddenDim));
    //     return new Dictionary<string, Call>
    //     {
    //         { "finalHiddenStates", finalHiddenStates },
    //         { "routerLogits", routerLogits },
    //     };
    // }
    private AttentionDimKind[] GetPagedAttentionTensorLayout(IPagedAttentionConfig config, AttentionDimKind[] sourceLayout)
    {
        if (UsesSourcePagedAttentionTensorLayout(config))
        {
            return sourceLayout;
        }

        return [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq];
    }

    private bool UsesSourcePagedAttentionTensorLayout(IPagedAttentionConfig config)
    {
        return !config.KeyCacheLayout.SequenceEqual(config.ValueCacheLayout) ||
            config.GetVectorizedAxes(AttentionCacheKind.Value).Contains(PagedKVCacheDimKind.BlockSize);
    }

    private Expr TransposeIfNeeded(Expr input, IReadOnlyList<int> perm)
    {
        return perm.Select((axis, index) => axis == index).All(x => x) ? input : IR.F.Tensors.Transpose(input, perm.ToArray());
    }
}
