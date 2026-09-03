// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualBasic;
using Nncase;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Passes.Distributed;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CoreTest;

public class UnitTestCodeGenUtil
{
    [Fact]
    public void TestCodeGenUtil()
    {
        string tempPath = Path.GetTempPath() + Guid.NewGuid().ToString();
        Assert.NotEqual(tempPath, CodeGenUtil.GetTempFileName());
    }

    [Fact]
    public void TestStructToBytes()
    {
        var num = new[] { new byte[] { 1, 2, 3 }, new byte[] { 2, 3, 4 }, new byte[] { 3, 4, 5 } };
        Assert.Throws<ArgumentException>(() => CodeGenUtil.StructToBytes(num));
    }

    [Fact]
    public void TestTypeSerializerLowersDistributedTypeToRuntimeTensorAbi()
    {
        var tensorType = new TensorType(DataTypes.BFloat16, [1, 5120]);
        var distributedType = new DistributedType(
            tensorType,
            [SBP.B, SBP.B],
            new Placement([4, 8], "yx", "bb"));
        using var expected = new MemoryStream();
        using var actual = new MemoryStream();
        using (var writer = new BinaryWriter(expected, System.Text.Encoding.UTF8, true))
        {
            TypeSerializer.Serialize(writer, tensorType);
        }

        using (var writer = new BinaryWriter(actual, System.Text.Encoding.UTF8, true))
        {
            TypeSerializer.Serialize(writer, distributedType);
        }

        Assert.Equal(expected.ToArray(), actual.ToArray());
    }
}
