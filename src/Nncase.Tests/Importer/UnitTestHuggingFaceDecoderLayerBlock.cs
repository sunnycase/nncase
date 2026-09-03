// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Importer;
using Xunit;

namespace Nncase.Tests.ImporterTest;

public sealed class UnitTestHuggingFaceDecoderLayerBlock
{
    [Fact]
    public void HomogeneousDecoderUsesOneLayerBlock()
    {
        Assert.Equal(1, TestModel.GetBlockLength("dense", "dense", "dense", "dense"));
    }

    [Fact]
    public void HybridDecoderUsesSmallestCompleteRepeatingBlock()
    {
        Assert.Equal(
            4,
            TestModel.GetBlockLength(
                "linear",
                "linear",
                "linear",
                "full",
                "linear",
                "linear",
                "linear",
                "full"));
    }

    [Fact]
    public void TruncatedHybridDecoderDoesNotReuseAnIncompleteBlock()
    {
        Assert.Equal(
            5,
            TestModel.GetBlockLength("linear", "linear", "linear", "full", "linear"));
    }

    [Fact]
    public void FunctionReuseMustBeConsistentAcrossDecoderStack()
    {
        Assert.Equal(0, TestModel.GetBlockLength(null, null));
        Assert.Throws<InvalidOperationException>(() => TestModel.GetBlockLength("dense", null));
    }

    private sealed class TestModel : HuggingFaceModel
    {
        public static int GetBlockLength(params string?[] structureKeys)
            => GetDecoderLayerBlockLength(structureKeys);
    }
}
