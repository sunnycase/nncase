// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Xunit;

namespace Nncase.Tests.CostModelTest;

public sealed class UnitTestCost
{
    [Fact]
    public void TestScoreOverlapsCpuAndMemoryThenAddsSynchronization()
    {
        var cost = new Cost
        {
            [CostFactorNames.CPUCycles] = 100,
            [CostFactorNames.ChipGlobalMemoryLoadBytes] = 40,
            [CostFactorNames.ChipGlobalMemoryStoreBytes] = 70,
            [CostFactorNames.GridSynchronization] = 5,
            [CostFactorNames.Comm] = 7,
        };

        Assert.Equal((UInt128)122, cost.Score);
    }

    [Fact]
    public void TestScoreKeepsUnknownFactorsAdditive()
    {
        var cost = new Cost
        {
            [CostFactorNames.CPUCycles] = 100,
            [CostFactorNames.ChipGlobalMemoryLoadBytes] = 10,
            [CostFactorNames.ChipGlobalMemoryStoreBytes] = 20,
            ["Custom"] = 3,
        };

        Assert.Equal((UInt128)103, cost.Score);
    }
}
