﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using LanguageExt.UnsafeValueAccess;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitConv2DTranspose(NodeProto op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            var group = GetIntAttribute(op, "group", 1);
            var bias = GetBias(op, weights, true, group);
            var strides = GetStrideAttribute(op).ToArray<long>().ToList();
            var dilation = GetDilationsAttribute(op).ToList();
            var autoPad = GetStringAttribute(op, "auto_pad", "NOTSET");

            var isConv1D = IsConv1D(weights);
            if (isConv1D)
            {
                dilation.Add(1);
                strides.Add(1);
                input = To4D(input);
                weights = To4D(weights);
            }

            var outputPadding = GetIntsAttribute(op, "output_padding", new[] { 0, 0 });
            var pads = AutoPad(op, autoPad, input, weights, strides.ToArray<long>(), dilation.ToArray<long>(), isConv1D);
            pads.InferenceType();

            var outShape = GetOptionIntsAttribute(op, "output_shape")
                .Match(
                    o => Tensor.From<long>(o),
                    () => GetOutputShape(
                        input,
                        weights,
                        strides.ToArray(),
                        outputPadding,
                        pads,
                        dilation.ToArray(),
                        autoPad,
                        group));

            weights = IR.F.Tensors.Transpose(weights, new[] { 1, 0, 2, 3 });
            var conv = F.NN.Conv2DTranspose(
                input,
                weights,
                bias,
                outShape,
                strides.ToArray(),
                pads,
                Tensor.From<long>(outputPadding),
                Tensor.From<long>(dilation.ToArray()),
                PadMode.Constant,
                group);

            if (isConv1D)
            {
                conv = Squeeze(conv, new[] { 3 });
            }

            return conv;
        }

        private Expr ComputeOutSize(Expr inputSize, Expr weightSize, long[] strides, long[] outPaddings, Expr paddings, long[] dilations, int offset)
        {
            return (strides[offset] * (inputSize - 1L))
                + outPaddings[offset]
                + (((weightSize - 1L)
                * dilations[offset]) + 1L) - paddings[offset][0] - paddings[offset][1];
        }

        private Expr GetOutputShape(Expr input, Expr weights, long[] strides, long[] outPadding, Expr paddings, long[] dilations, string autoPad, long group)
        {
            var iN = Util.ShapeIndex(input, 0);
            _ = Util.ShapeIndex(input, 1);
            var (iH, iW) = Util.GetHW(input);
            var oc = Util.ShapeIndex(weights, 1) * group;

            // var ic = Util.ShapeIndex(weights, 1);
            var (wH, wW) = Util.GetHW(weights);
            var outShape = new List<Expr>();
            outShape.Add(iN);
            outShape.Add(oc);
            if (autoPad is "SAME_UPPER" or "SAME_LOWER")
            {
                outShape.Add(iH * strides[0]);
                outShape.Add(iW * strides[1]);
            }
            else
            {
                outShape.Add(ComputeOutSize(iH, wH, strides, outPadding, paddings, dilations, 0));
                outShape.Add(ComputeOutSize(iW, wW, strides, outPadding, paddings, dilations, 1));
            }

            return F.Tensors.Stack(new IR.Tuple(CollectionsMarshal.AsSpan(outShape)), 0);
        }
    }
}
