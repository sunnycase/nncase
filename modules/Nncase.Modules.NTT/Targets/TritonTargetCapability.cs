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
    public TritonDotInstructionCapability(string name, int m, int n, int k, double instructionsPerCyclePerCta, bool isSupported = true)
    {
        Name = name;
        M = m;
        N = n;
        K = k;
        InstructionsPerCyclePerCta = instructionsPerCyclePerCta;
        IsSupported = isSupported;
    }

    public string Name { get; init; }

    public int M { get; init; }

    public int N { get; init; }

    public int K { get; init; }

    public double InstructionsPerCyclePerCta { get; init; }

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

    public double GlobalMemoryBandwidthGBps { get; init; } = 1555.0;

    public double GlobalMemoryElementsPerCyclePerCta { get; init; } = 32.0;

    public double GlobalMemoryElementSizeBytes { get; init; } = 4.0;

    public double GlobalMemoryEfficiency { get; init; } = 1.0;

    public double ElementwiseElementsPerCyclePerCta { get; init; } = 128.0;

    public double SimtFmaPerCyclePerCta { get; init; } = 64.0;

    public double FixedOverheadCycles { get; init; }

    public double SynchronizationCyclesPerEvent { get; init; } = 2200.0;

    public bool UseTensorCoresForFloat32 { get; init; } = true;

    public TritonDotInstructionCapability Mma { get; init; } = DefaultMma;

    public TritonDotInstructionCapability Wgmma { get; init; } = DefaultWgmma with { IsSupported = false };

    public int ComputeCapability => (ComputeCapabilityMajor * 10) + ComputeCapabilityMinor;

    public double EffectiveGlobalMemoryElementsPerCyclePerCta => GlobalMemoryElementsPerCyclePerCta > 0
        ? GlobalMemoryElementsPerCyclePerCta
        : (GlobalMemoryBandwidthGBps / Math.Max(1.0e-6, ClockRateGHz) / Math.Max(1.0e-6, GlobalMemoryElementSizeBytes)) * GlobalMemoryEfficiency;

    public double SynchronizationLatencyUs => SynchronizationCyclesPerEvent / Math.Max(1.0e-6, ClockRateGHz) / 1000.0;

    public double SynchronizationCycles => Math.Max(0.0, SynchronizationCyclesPerEvent);

    public static TritonTargetCapability ForComputeCapability(int major, int minor)
    {
        var supportsWgmma = major >= 9;
        return new TritonTargetCapability
        {
            ComputeCapabilityMajor = major,
            ComputeCapabilityMinor = minor,
            MultiprocessorCount = supportsWgmma ? 132 : 108,
            ClockRateGHz = supportsWgmma ? 1.8 : 1.4,
            GlobalMemoryBandwidthGBps = supportsWgmma ? 3000.0 : 1555.0,
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

        if (values.TryGetValue("mem_bw_gbps", out var memoryBandwidth) || values.TryGetValue("hbm_gbps", out memoryBandwidth) || values.TryGetValue("bandwidth_gbps", out memoryBandwidth))
        {
            capability = capability with { GlobalMemoryBandwidthGBps = ParseDouble(memoryBandwidth, nameof(GlobalMemoryBandwidthGBps)) };
        }

        if (values.TryGetValue("memory_epc", out var memoryEpc) || values.TryGetValue("mem_epc", out memoryEpc))
        {
            capability = capability with { GlobalMemoryElementsPerCyclePerCta = ParseDouble(memoryEpc, nameof(GlobalMemoryElementsPerCyclePerCta), allowZero: true) };
        }

        if (values.TryGetValue("memory_element_bytes", out var memoryElementBytes) || values.TryGetValue("mem_element_bytes", out memoryElementBytes))
        {
            capability = capability with { GlobalMemoryElementSizeBytes = ParseDouble(memoryElementBytes, nameof(GlobalMemoryElementSizeBytes)) };
        }

        if (values.TryGetValue("memory_efficiency", out var memoryEfficiency) || values.TryGetValue("mem_efficiency", out memoryEfficiency))
        {
            capability = capability with { GlobalMemoryEfficiency = ParseDouble(memoryEfficiency, nameof(GlobalMemoryEfficiency)) };
        }

        if (values.TryGetValue("elementwise_epc", out var elementwiseEpc))
        {
            capability = capability with { ElementwiseElementsPerCyclePerCta = ParseDouble(elementwiseEpc, nameof(ElementwiseElementsPerCyclePerCta)) };
        }

        if (values.TryGetValue("simt_fma_per_cycle", out var simtFma))
        {
            capability = capability with { SimtFmaPerCyclePerCta = ParseDouble(simtFma, nameof(SimtFmaPerCyclePerCta)) };
        }

        if (values.TryGetValue("overhead_cycles", out var overheadCycles) || values.TryGetValue("fixed_overhead_cycles", out overheadCycles))
        {
            capability = capability with { FixedOverheadCycles = ParseDouble(overheadCycles, nameof(FixedOverheadCycles), allowZero: true) };
        }

        if (values.TryGetValue("sync_cycles", out var syncCycles) || values.TryGetValue("synchronization_cycles", out syncCycles))
        {
            capability = capability with { SynchronizationCyclesPerEvent = ParseDouble(syncCycles, nameof(SynchronizationCyclesPerEvent), allowZero: true) };
        }

        if (values.TryGetValue("sync_us", out var syncUs) || values.TryGetValue("sync_latency_us", out syncUs) || values.TryGetValue("synchronization_us", out syncUs))
        {
            var cycles = ParseDouble(syncUs, nameof(SynchronizationLatencyUs), allowZero: true) * Math.Max(1.0e-6, capability.ClockRateGHz) * 1000.0;
            capability = capability with { SynchronizationCyclesPerEvent = cycles };
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
            capability = capability with { Mma = capability.Mma with { InstructionsPerCyclePerCta = ParseDouble(mmaIpc, nameof(TritonDotInstructionCapability.InstructionsPerCyclePerCta)) } };
        }

        if (values.TryGetValue("wgmma", out var wgmmaTile))
        {
            capability = capability with { Wgmma = ParseInstructionTile(capability.Wgmma, wgmmaTile) with { IsSupported = true } };
        }

        if (values.TryGetValue("wgmma_ipc", out var wgmmaIpc))
        {
            capability = capability with { Wgmma = capability.Wgmma with { InstructionsPerCyclePerCta = ParseDouble(wgmmaIpc, nameof(TritonDotInstructionCapability.InstructionsPerCyclePerCta)) } };
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
            $"cc={ComputeCapability},num_sms={MultiprocessorCount},clock_ghz={ClockRateGHz},mem_bw_gbps={GlobalMemoryBandwidthGBps},memory_epc={EffectiveGlobalMemoryElementsPerCyclePerCta},memory_element_bytes={GlobalMemoryElementSizeBytes},memory_efficiency={GlobalMemoryEfficiency},sync_cycles={SynchronizationCyclesPerEvent},sync_us={SynchronizationLatencyUs},mma={Mma.M}x{Mma.N}x{Mma.K},mma_ipc={Mma.InstructionsPerCyclePerCta},wgmma={Wgmma.M}x{Wgmma.N}x{Wgmma.K},wgmma_ipc={Wgmma.InstructionsPerCyclePerCta},wgmma_supported={Wgmma.IsSupported}");
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
