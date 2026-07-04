// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace Nncase.Targets;

/// <summary>
/// Triton dot instruction capability.
/// </summary>
public sealed record TritonDotInstructionCapability
{
    public TritonDotInstructionCapability(string name, int m, int n, int k, double instructionsPerCyclePerBlock, bool isSupported = true)
    {
        Name = name;
        M = m;
        N = n;
        K = k;
        InstructionsPerCyclePerBlock = instructionsPerCyclePerBlock;
        IsSupported = isSupported;
    }

    public string Name { get; init; }

    public int M { get; init; }

    public int N { get; init; }

    public int K { get; init; }

    public double InstructionsPerCyclePerBlock { get; init; }

    public bool IsSupported { get; init; }
}

/// <summary>
/// Triton backend hardware capability used by target cost models.
/// </summary>
public sealed record TritonTargetCapability
{
    public static readonly TritonDotInstructionCapability DefaultMma = new("mma", 16, 8, 16, 4.0);

    public static readonly TritonDotInstructionCapability DefaultWgmma = new("wgmma", 64, 8, 16, 8.0);

    public static readonly TritonTargetCapability Default = ForComputeCapability(8, 0);

    public int ComputeCapabilityMajor { get; init; } = 8;

    public int ComputeCapabilityMinor { get; init; }

    public int MultiprocessorCount { get; init; } = 108;

    public int WarpSize { get; init; } = 32;

    public double ClockRateGHz { get; init; } = 1.4;

    public double ChipGlobalMemoryBandwidthGBps { get; init; } = 1555.0;

    public double ChipGlobalMemoryBytesPerCycle { get; init; }

    public double ChipGlobalMemoryEfficiency { get; init; } = 1.0;

    public double BlockLocalMemoryBytesPerCyclePerBlock { get; init; } = 512.0;

    public double BlockLocalMemoryEfficiency { get; init; } = 1.0;

    public double ElementwiseElementsPerCyclePerBlock { get; init; } = 128.0;

    public double SimtFmaPerCyclePerBlock { get; init; } = 64.0;

    public double FixedOverheadCycles { get; init; }

    public double BlockSynchronizationCyclesPerEvent { get; init; } = 25.0;

    public double GridSynchronizationCyclesPerEvent { get; init; } = 2200.0;

    public bool UseTensorCoresForFloat32 { get; init; } = true;

    public TritonDotInstructionCapability Mma { get; init; } = DefaultMma;

    public TritonDotInstructionCapability Wgmma { get; init; } = DefaultWgmma with { IsSupported = false };

    public int ComputeCapability => (ComputeCapabilityMajor * 10) + ComputeCapabilityMinor;

    public double EffectiveChipGlobalMemoryBytesPerCycle => ChipGlobalMemoryBytesPerCycle > 0
        ? ChipGlobalMemoryBytesPerCycle
        : (ChipGlobalMemoryBandwidthGBps / Math.Max(1.0e-6, ClockRateGHz)) * ChipGlobalMemoryEfficiency;

    public double EffectiveBlockLocalMemoryBytesPerCyclePerBlock => BlockLocalMemoryBytesPerCyclePerBlock * BlockLocalMemoryEfficiency;

    public double GridSynchronizationLatencyUs => GridSynchronizationCyclesPerEvent / Math.Max(1.0e-6, ClockRateGHz) / 1000.0;

    public double BlockSynchronizationCycles => Math.Max(0.0, BlockSynchronizationCyclesPerEvent);

    public double GridSynchronizationCycles => Math.Max(0.0, GridSynchronizationCyclesPerEvent);

    public static TritonTargetCapability ForComputeCapability(int major, int minor)
    {
        var supportsWgmma = major >= 9;
        return new TritonTargetCapability
        {
            ComputeCapabilityMajor = major,
            ComputeCapabilityMinor = minor,
            MultiprocessorCount = supportsWgmma ? 132 : 108,
            ClockRateGHz = supportsWgmma ? 1.8 : 1.4,
            ChipGlobalMemoryBandwidthGBps = supportsWgmma ? 3000.0 : 1555.0,
            Wgmma = DefaultWgmma with { IsSupported = supportsWgmma },
        };
    }

    public static TritonTargetCapability Parse(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return Default;
        }

        text = text.Trim();
        if (TryParseComputeCapability(text, out var major, out var minor))
        {
            return ForComputeCapability(major, minor);
        }

        var values = ParseKeyValues(text);
        var capability = values.TryGetValue("cc", out var cc) || values.TryGetValue("capability", out cc) || values.TryGetValue("sm", out cc)
            ? Parse(cc)
            : Default;

        if (values.TryGetValue("num_sms", out var numSms) || values.TryGetValue("sms", out numSms) || values.TryGetValue("multiprocessor_count", out numSms))
        {
            capability = capability with { MultiprocessorCount = ParseInt(numSms, nameof(MultiprocessorCount)) };
        }

        if (values.TryGetValue("warp_size", out var warpSize))
        {
            capability = capability with { WarpSize = ParseInt(warpSize, nameof(WarpSize)) };
        }

        if (values.TryGetValue("clock_ghz", out var clockGhz) || values.TryGetValue("clock", out clockGhz))
        {
            capability = capability with { ClockRateGHz = ParseDouble(clockGhz, nameof(ClockRateGHz)) };
        }

        if (values.TryGetValue("chip_mem_bw_gbps", out var memoryBandwidth) || values.TryGetValue("mem_bw_gbps", out memoryBandwidth) || values.TryGetValue("hbm_gbps", out memoryBandwidth) || values.TryGetValue("bandwidth_gbps", out memoryBandwidth))
        {
            capability = capability with { ChipGlobalMemoryBandwidthGBps = ParseDouble(memoryBandwidth, nameof(ChipGlobalMemoryBandwidthGBps)) };
        }

        if (values.TryGetValue("chip_memory_bpc", out var memoryBpc) || values.TryGetValue("memory_bpc", out memoryBpc) || values.TryGetValue("mem_bpc", out memoryBpc))
        {
            capability = capability with { ChipGlobalMemoryBytesPerCycle = ParseDouble(memoryBpc, nameof(ChipGlobalMemoryBytesPerCycle), allowZero: true) };
        }

        if (values.TryGetValue("chip_memory_efficiency", out var memoryEfficiency) || values.TryGetValue("memory_efficiency", out memoryEfficiency) || values.TryGetValue("mem_efficiency", out memoryEfficiency))
        {
            capability = capability with { ChipGlobalMemoryEfficiency = ParseDouble(memoryEfficiency, nameof(ChipGlobalMemoryEfficiency)) };
        }

        if (values.TryGetValue("block_memory_bpc", out var blockMemoryBpc) || values.TryGetValue("block_mem_bpc", out blockMemoryBpc) || values.TryGetValue("blocklocal_memory_bpc", out blockMemoryBpc))
        {
            capability = capability with { BlockLocalMemoryBytesPerCyclePerBlock = ParseDouble(blockMemoryBpc, nameof(BlockLocalMemoryBytesPerCyclePerBlock)) };
        }

        if (values.TryGetValue("block_memory_efficiency", out var blockMemoryEfficiency) || values.TryGetValue("block_mem_efficiency", out blockMemoryEfficiency) || values.TryGetValue("blocklocal_memory_efficiency", out blockMemoryEfficiency))
        {
            capability = capability with { BlockLocalMemoryEfficiency = ParseDouble(blockMemoryEfficiency, nameof(BlockLocalMemoryEfficiency)) };
        }

        if (values.TryGetValue("elementwise_epc", out var elementwiseEpc))
        {
            capability = capability with { ElementwiseElementsPerCyclePerBlock = ParseDouble(elementwiseEpc, nameof(ElementwiseElementsPerCyclePerBlock)) };
        }

        if (values.TryGetValue("simt_fma_per_cycle", out var simtFma))
        {
            capability = capability with { SimtFmaPerCyclePerBlock = ParseDouble(simtFma, nameof(SimtFmaPerCyclePerBlock)) };
        }

        if (values.TryGetValue("overhead_cycles", out var overheadCycles) || values.TryGetValue("fixed_overhead_cycles", out overheadCycles))
        {
            capability = capability with { FixedOverheadCycles = ParseDouble(overheadCycles, nameof(FixedOverheadCycles), allowZero: true) };
        }

        if (values.TryGetValue("block_sync_cycles", out var blockSyncCycles) || values.TryGetValue("block_synchronization_cycles", out blockSyncCycles))
        {
            capability = capability with { BlockSynchronizationCyclesPerEvent = ParseDouble(blockSyncCycles, nameof(BlockSynchronizationCyclesPerEvent), allowZero: true) };
        }

        if (values.TryGetValue("sync_cycles", out var syncCycles) || values.TryGetValue("grid_sync_cycles", out syncCycles) || values.TryGetValue("synchronization_cycles", out syncCycles))
        {
            capability = capability with { GridSynchronizationCyclesPerEvent = ParseDouble(syncCycles, nameof(GridSynchronizationCyclesPerEvent), allowZero: true) };
        }

        if (values.TryGetValue("block_sync_us", out var blockSyncUs) || values.TryGetValue("block_synchronization_us", out blockSyncUs))
        {
            var cycles = ParseDouble(blockSyncUs, nameof(BlockSynchronizationCyclesPerEvent), allowZero: true) * Math.Max(1.0e-6, capability.ClockRateGHz) * 1000.0;
            capability = capability with { BlockSynchronizationCyclesPerEvent = cycles };
        }

        if (values.TryGetValue("sync_us", out var syncUs) || values.TryGetValue("grid_sync_us", out syncUs) || values.TryGetValue("sync_latency_us", out syncUs) || values.TryGetValue("synchronization_us", out syncUs))
        {
            var cycles = ParseDouble(syncUs, nameof(GridSynchronizationLatencyUs), allowZero: true) * Math.Max(1.0e-6, capability.ClockRateGHz) * 1000.0;
            capability = capability with { GridSynchronizationCyclesPerEvent = cycles };
        }

        if (values.TryGetValue("tf32", out var tf32))
        {
            capability = capability with { UseTensorCoresForFloat32 = ParseBool(tf32, nameof(UseTensorCoresForFloat32)) };
        }

        if (values.TryGetValue("mma", out var mmaTile))
        {
            capability = capability with { Mma = ParseInstructionTile(capability.Mma, mmaTile) };
        }

        if (values.TryGetValue("mma_ipc", out var mmaIpc))
        {
            capability = capability with { Mma = capability.Mma with { InstructionsPerCyclePerBlock = ParseDouble(mmaIpc, nameof(TritonDotInstructionCapability.InstructionsPerCyclePerBlock)) } };
        }

        if (values.TryGetValue("wgmma", out var wgmmaTile))
        {
            capability = capability with { Wgmma = ParseInstructionTile(capability.Wgmma, wgmmaTile) with { IsSupported = true } };
        }

        if (values.TryGetValue("wgmma_ipc", out var wgmmaIpc))
        {
            capability = capability with { Wgmma = capability.Wgmma with { InstructionsPerCyclePerBlock = ParseDouble(wgmmaIpc, nameof(TritonDotInstructionCapability.InstructionsPerCyclePerBlock)) } };
        }

        if (values.TryGetValue("wgmma_supported", out var wgmmaSupported))
        {
            capability = capability with { Wgmma = capability.Wgmma with { IsSupported = ParseBool(wgmmaSupported, nameof(TritonDotInstructionCapability.IsSupported)) } };
        }

        return capability;
    }

    public override string ToString()
    {
        return FormattableString.Invariant(
            $"cc={ComputeCapability},num_sms={MultiprocessorCount},clock_ghz={ClockRateGHz},chip_mem_bw_gbps={ChipGlobalMemoryBandwidthGBps},chip_memory_bpc={EffectiveChipGlobalMemoryBytesPerCycle},chip_memory_efficiency={ChipGlobalMemoryEfficiency},block_memory_bpc={EffectiveBlockLocalMemoryBytesPerCyclePerBlock},block_memory_efficiency={BlockLocalMemoryEfficiency},block_sync_cycles={BlockSynchronizationCyclesPerEvent},sync_cycles={GridSynchronizationCyclesPerEvent},sync_us={GridSynchronizationLatencyUs},mma={Mma.M}x{Mma.N}x{Mma.K},mma_ipc={Mma.InstructionsPerCyclePerBlock},wgmma={Wgmma.M}x{Wgmma.N}x{Wgmma.K},wgmma_ipc={Wgmma.InstructionsPerCyclePerBlock},wgmma_supported={Wgmma.IsSupported}");
    }

    private static Dictionary<string, string> ParseKeyValues(string text)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var separator = item.IndexOf('=', StringComparison.Ordinal);
            if (separator <= 0 || separator == item.Length - 1)
            {
                throw new FormatException($"Invalid Triton capability item '{item}'. Expected key=value.");
            }

            values[item[..separator].Trim()] = item[(separator + 1)..].Trim();
        }

        return values;
    }

    private static TritonDotInstructionCapability ParseInstructionTile(TritonDotInstructionCapability instruction, string value)
    {
        var parts = value.Split('x', 'X');
        if (parts.Length != 3)
        {
            throw new FormatException($"Invalid Triton {instruction.Name} tile '{value}'. Expected MxNxK.");
        }

        return instruction with
        {
            M = ParseInt(parts[0], $"{instruction.Name}.M"),
            N = ParseInt(parts[1], $"{instruction.Name}.N"),
            K = ParseInt(parts[2], $"{instruction.Name}.K"),
        };
    }

    private static bool TryParseComputeCapability(string text, out int major, out int minor)
    {
        text = text.Trim().ToUpperInvariant();
        if (text.StartsWith("CUDA:", StringComparison.Ordinal))
        {
            text = text["cuda:".Length..];
        }

        if (text.StartsWith("SM", StringComparison.Ordinal))
        {
            text = text["sm".Length..];
        }

        if (text.Contains('.', StringComparison.Ordinal))
        {
            var parts = text.Split('.');
            if (parts.Length == 2
                && int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out major)
                && int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out minor))
            {
                return true;
            }
        }
        else if (int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var cc))
        {
            major = cc / 10;
            minor = cc % 10;
            return major > 0;
        }

        major = 0;
        minor = 0;
        return false;
    }

    private static int ParseInt(string value, string name)
    {
        if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var result) && result > 0)
        {
            return result;
        }

        throw new FormatException($"Invalid Triton capability integer '{name}={value}'.");
    }

    private static double ParseDouble(string value, string name, bool allowZero = false)
    {
        if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var result)
            && double.IsFinite(result)
            && (allowZero ? result >= 0 : result > 0))
        {
            return result;
        }

        throw new FormatException($"Invalid Triton capability number '{name}={value}'.");
    }

    private static bool ParseBool(string value, string name)
    {
        if (bool.TryParse(value, out var result))
        {
            return result;
        }

        if (value == "1")
        {
            return true;
        }

        if (value == "0")
        {
            return false;
        }

        throw new FormatException($"Invalid Triton capability boolean '{name}={value}'.");
    }
}
