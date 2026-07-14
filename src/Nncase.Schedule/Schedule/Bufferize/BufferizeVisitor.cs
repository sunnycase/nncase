// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Security.Cryptography;
using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Schedule.Bufferize;

public sealed class BufferizeVisitor : ExprRewriter
{
    private static readonly MemoryLocation[] WorkspaceLocations =
    [
        MemoryLocation.Data,
        MemoryLocation.ChipLocalData,
        MemoryLocation.BlockLocalData,
    ];

    private readonly IGrouping<string, PrimFunction> _functions;
    private readonly Dictionary<ReadOnlyDataKey, ValueRange<ulong>> _rdataRanges = new();
    private readonly Dictionary<ReadOnlyDataKey, ValueRange<ulong>> _chipLocalRdataRanges = new();
    private readonly Dictionary<PrimFunction, PrimFunction> _workspaceAbiFunctions = new(new ReferenceEqualityComparer<PrimFunction>());
    private long _currentRdataStart;
    private long _currentChipLocalRdataStart;
    private long _currentBlockLocalRdataStart;
    private int _dataBufferId;

    public BufferizeVisitor(IGrouping<string, PrimFunction> functions)
        : base(visitOtherFunctions: true)
    {
        _functions = functions;
    }

    public void Bufferize()
    {
        foreach (var func in _functions)
        {
            Rewrite(func);
        }
    }

    protected override BaseExpr RewriteLeafPrimFunction(PrimFunction func)
    {
        if (!func.SchedResult.IsScheduled)
        {
            (var buffers, var lifetimes) = new LifetimeCollector().Collect(func);
            var scheduleResult = BufferScheduler.Schedule(lifetimes, x => x switch
            {
                MemoryLocation.Rdata => new BufferScheduleOptions(_currentRdataStart),
                MemoryLocation.ChipLocalRdata => new BufferScheduleOptions(_currentChipLocalRdataStart),
                MemoryLocation.BlockLocalRdata => new BufferScheduleOptions(_currentBlockLocalRdataStart),
                _ => new BufferScheduleOptions(),
            });
            ReuseReadOnlyRDataResult(scheduleResult, MemoryLocation.Rdata, _rdataRanges);
            ReuseReadOnlyRDataResult(scheduleResult, MemoryLocation.ChipLocalRdata, _chipLocalRdataRanges);

            if (DumpScope.Current.IsEnabled(DumpFlags.Schedule))
            {
                using var dumpScope = new DumpScope(func.Name);
                DumpSchedule(buffers, scheduleResult);
            }

            AssignOutputResult(func, scheduleResult);
            AssignDataResult(func, scheduleResult);
            AssignChipLocalDataResult(func, scheduleResult);
            AssignBlockLocalDataResult(func, scheduleResult);
            AssignRdataResult(func, scheduleResult);
            AssignChipLocalRdataResult(func, scheduleResult);
            AssignBlockLocalRdataResult(func, scheduleResult);

            var bufferReplaces = scheduleResult.SelectMany(x => x.Value.Buffers).ToDictionary(ReferenceEqualityComparer.Instance);
            new BufferReplacer(bufferReplaces).Rewrite(func.Body);
            func.SchedResult.IsScheduled = true;
        }

        return func;
    }

    protected override BaseExpr RewriteLeafCall(Call expr)
    {
        if (expr.Target is not PrimFunction func)
        {
            return expr;
        }

        if (!func.SchedResult.IsScheduled)
        {
            throw new InvalidOperationException($"Function {func.Name} is not scheduled, please run BufferizePass first.");
        }

        var workspaceParameters = GetWorkspaceParameters(func);
        if (workspaceParameters.Length != 0)
        {
            ValidateWorkspaceParameters(func, workspaceParameters);
            if (expr.Arguments.Length == func.Parameters.Length)
            {
                return expr;
            }

            var nonWorkspaceParameterCount = func.Parameters.Length - workspaceParameters.Length;
            if (expr.Arguments.Length != nonWorkspaceParameterCount)
            {
                throw new InvalidOperationException(
                    $"Cannot bufferize call to {func.Name}: expected {nonWorkspaceParameterCount} arguments before " +
                    $"workspace lowering or {func.Parameters.Length} arguments after lowering, got {expr.Arguments.Length}.");
            }

            return expr.With(arguments: expr.Arguments.ToArray().Concat(CreateWorkspaceArguments(func)).ToArray());
        }

        if (expr.Arguments.Length != func.Parameters.Length)
        {
            throw new InvalidOperationException(
                $"Cannot bufferize call to {func.Name}: expected {func.Parameters.Length} arguments, got {expr.Arguments.Length}.");
        }

        var workspaceAbiFunction = GetOrCreateWorkspaceAbiFunction(func);
        var arguments = expr.Arguments.ToArray().Concat(CreateWorkspaceArguments(func)).ToArray();
        return expr.With(target: workspaceAbiFunction, arguments: arguments);
    }

    private PrimFunction GetOrCreateWorkspaceAbiFunction(PrimFunction func)
    {
        if (_workspaceAbiFunctions.TryGetValue(func, out var workspaceAbiFunction))
        {
            return workspaceAbiFunction;
        }

        var parameters = func.Parameters.ToArray().Concat(CreateWorkspaceParameters()).ToArray();
        workspaceAbiFunction = func.With(parameters: parameters);
        _workspaceAbiFunctions.Add(func, workspaceAbiFunction);
        return workspaceAbiFunction;
    }

    private BufferVar[] CreateWorkspaceParameters()
        => WorkspaceLocations
            .Select(location => new BufferVar(
                GetWorkspaceName(location),
                TensorType.Scalar(new PointerType(DataTypes.UInt8)),
                BufferVarRole.Workspace,
                location))
            .ToArray();

    private BaseExpr[] CreateWorkspaceArguments(PrimFunction func)
        => WorkspaceLocations.Select(location => (BaseExpr)CreateWorkspaceBuffer(func, location)).ToArray();

    private TIR.Buffer CreateWorkspaceBuffer(PrimFunction func, MemoryLocation location)
    {
        var size = location switch
        {
            MemoryLocation.Data => func.SchedResult.DataUsage,
            MemoryLocation.ChipLocalData => func.SchedResult.ChipLocalDataPoolSize,
            MemoryLocation.BlockLocalData => func.SchedResult.BlockLocalDataPoolSize,
            _ => throw new ArgumentOutOfRangeException(nameof(location), location, "Unsupported workspace memory location."),
        };
        var name = $"{GetWorkspaceName(location)}_{_dataBufferId++}";
        return T.CreateBuffer(
            new TensorType(DataTypes.UInt8, [checked((long)size)]),
            location,
            out _,
            name);
    }

    private BufferVar[] GetWorkspaceParameters(PrimFunction func)
        => func.Parameters
            .ToArray()
            .OfType<BufferVar>()
            .Where(parameter => parameter.Role == BufferVarRole.Workspace)
            .ToArray();

    private void ValidateWorkspaceParameters(PrimFunction func, IReadOnlyList<BufferVar> workspaceParameters)
    {
        if (workspaceParameters.Count != WorkspaceLocations.Length)
        {
            throw new InvalidOperationException(
                $"PrimFunction {func.Name} must have exactly {WorkspaceLocations.Length} workspace parameters, " +
                $"got {workspaceParameters.Count}.");
        }

        var parameters = func.Parameters.ToArray();
        for (int i = 0; i < WorkspaceLocations.Length; i++)
        {
            var parameterIndex = parameters.Length - WorkspaceLocations.Length + i;
            if (!ReferenceEquals(parameters[parameterIndex], workspaceParameters[i]))
            {
                throw new InvalidOperationException(
                    $"PrimFunction {func.Name} workspace parameters must be the final ABI parameters.");
            }

            if (workspaceParameters[i].Location != WorkspaceLocations[i])
            {
                throw new InvalidOperationException(
                    $"PrimFunction {func.Name} workspace parameter {workspaceParameters[i].Name} has location " +
                    $"{workspaceParameters[i].Location}, expected {WorkspaceLocations[i]}.");
            }
        }
    }

    private string GetWorkspaceName(MemoryLocation location)
        => location switch
        {
            MemoryLocation.Data => "data",
            MemoryLocation.ChipLocalData => "chip_local_data",
            MemoryLocation.BlockLocalData => "block_local_data",
            _ => throw new ArgumentOutOfRangeException(nameof(location), location, "Unsupported workspace memory location."),
        };

    private void AssignOutputResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.Output, out var dataResult))
        {
            func.SchedResult.OutputAlign = Math.Max(8, (ulong)dataResult.Alignment);
            func.SchedResult.OutputUsage = MathUtility.AlignUp((ulong)dataResult.MemoryPoolEnd, func.SchedResult.OutputAlign);
        }
    }

    private void AssignDataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.Data, out var dataResult))
        {
            func.SchedResult.DataAlign = Math.Max(8, (ulong)dataResult.Alignment);
            func.SchedResult.DataUsage = MathUtility.AlignUp((ulong)dataResult.MemoryPoolEnd, func.SchedResult.DataAlign);
        }
    }

    private void AssignChipLocalDataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.ChipLocalData, out var chipLocalDataResult))
        {
            func.SchedResult.DataAlign = Math.Max(8, (ulong)chipLocalDataResult.Alignment);
            func.SchedResult.ChipLocalDataPoolSize = MathUtility.AlignUp((ulong)chipLocalDataResult.MemoryPoolEnd, func.SchedResult.DataAlign);
        }
    }

    private void AssignBlockLocalDataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.BlockLocalData, out var blockLocalDataResult))
        {
            func.SchedResult.DataAlign = Math.Max(8, (ulong)blockLocalDataResult.Alignment);
            func.SchedResult.BlockLocalDataPoolSize = MathUtility.AlignUp((ulong)blockLocalDataResult.MemoryPoolEnd, func.SchedResult.DataAlign);
        }
    }

    private void AssignRdataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.Rdata, out var rdataResult))
        {
            foreach ((var buffer, var lifetime) in rdataResult.Buffers)
            {
                var constValue = (Const)((Call)buffer.Start)[IR.Buffers.AddressOf.Input];
                var range = new ValueRange<ulong>((ulong)lifetime.Memory.Start, (ulong)lifetime.Memory.Stop);
                func.SchedResult.Rdatas.Add(constValue, range);
            }

            _currentRdataStart = GetReadOnlyRDataEnd(_rdataRanges);
        }
    }

    private void AssignChipLocalRdataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.ChipLocalRdata, out var chipLocalRdataResult))
        {
            foreach ((var buffer, var lifetime) in chipLocalRdataResult.Buffers)
            {
                var constValue = (Const)((Call)buffer.Start)[IR.Buffers.AddressOf.Input];
                var range = new ValueRange<ulong>((ulong)lifetime.Memory.Start, (ulong)lifetime.Memory.Stop);
                func.SchedResult.ChipLocalRdatas.Add(constValue, range);
            }

            _currentChipLocalRdataStart = GetReadOnlyRDataEnd(_chipLocalRdataRanges);
        }
    }

    private void AssignBlockLocalRdataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.BlockLocalRdata, out var blockLocalRdataResult))
        {
            foreach ((var buffer, var lifetime) in blockLocalRdataResult.Buffers)
            {
                var constValue = (Const)((Call)buffer.Start)[IR.Buffers.AddressOf.Input];
                var range = new ValueRange<ulong>((ulong)lifetime.Memory.Start, (ulong)lifetime.Memory.Stop);
                func.SchedResult.BlockLocalRdatas.Add(constValue, range);
            }

            _currentBlockLocalRdataStart = blockLocalRdataResult.MemoryPoolEnd;
        }
    }

    private void ReuseReadOnlyRDataResult(
        IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult,
        MemoryLocation memoryLocation,
        Dictionary<ReadOnlyDataKey, ValueRange<ulong>> ranges)
    {
        if (!scheduleResult.TryGetValue(memoryLocation, out var result))
        {
            return;
        }

        foreach ((var buffer, var lifetime) in result.Buffers)
        {
            var constValue = GetAddressOfConst(buffer.Start);
            var key = GetReadOnlyDataKey(constValue);
            if (ranges.TryGetValue(key, out var existingRange))
            {
                lifetime.Memory = new Interval(checked((long)existingRange.Min), checked((long)existingRange.Max));
                continue;
            }

            ranges.Add(key, new ValueRange<ulong>((ulong)lifetime.Memory.Start, (ulong)lifetime.Memory.Stop));
        }
    }

    private long GetReadOnlyRDataEnd(Dictionary<ReadOnlyDataKey, ValueRange<ulong>> ranges)
        => ranges.Count == 0 ? 0L : checked((long)ranges.Values.Max(range => range.Max));

    private Const GetAddressOfConst(Expr start)
    {
        if (start is Call call && call.Target is IR.Buffers.AddressOf && call[IR.Buffers.AddressOf.Input] is Const constValue)
        {
            return constValue;
        }

        throw new InvalidOperationException($"Bufferized readonly data expects AddressOf(Const), got {start}.");
    }

    private ReadOnlyDataKey GetReadOnlyDataKey(Const constValue)
    {
        if (constValue is not TensorConst tensorConst)
        {
            throw new NotSupportedException($"Bufferized readonly data only supports TensorConst, got {constValue.GetType().Name}.");
        }

        var tensor = tensorConst.Value;
        return new(
            tensor.ElementType.GetDisplayName(),
            string.Join(",", tensor.Dimensions.ToArray()),
            string.Join(",", tensor.Strides.ToArray()),
            tensor.BytesBuffer.Length,
            Convert.ToHexString(SHA256.HashData(tensor.BytesBuffer)));
    }

    private void DumpSchedule(TIR.Buffer[] buffers, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        foreach (var group in buffers.GroupBy(x => x.MemSpan.Buffer.Location))
        {
            if (!scheduleResult.TryGetValue(group.Key, out var schedule))
            {
                continue;
            }

            using var wr = new StreamWriter(DumpScope.Current.OpenFile($"{group.Key}.py"), Encoding.UTF8);
            wr.Write(@"from bokeh.models import ColumnDataSource, HoverTool, SingleIntervalTicker, SaveTool, WheelZoomTool, WheelPanTool, ResetTool
from bokeh.palettes import Category20_20 as palette
from bokeh.plotting import figure, show, save
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import List

@dataclass
class Interval():
  start: int
  end: int
  def __str__(self) -> str:
    return f'(start: {self.start}, end {self.end}, size {self.end - self.start})'

class ConstraintsMode(Enum):
  No = 0
  Channel = 1

@dataclass
class ScheduledBuffer():
  name: str
  number: int
  time_interval: Interval
  mem_interval: Interval
  constraints: ConstraintsMode
  shape: List[str]
  stride: List[int]
  inplace: bool

colors = itertools.cycle(palette)

buffers = [
");
            int bufferId = 0;
            foreach (var buffer in group)
            {
                var lifetime = schedule.Buffers[buffer.MemSpan.Buffer];
                var dims = new RankedShape(buffer.Dimensions).Select(x => $"'{x}'");
                var strides = new RankedShape(buffer.Strides).Select(x => $"'{x}'");
                wr.WriteLine($"ScheduledBuffer('{buffer.Name}', {bufferId}, {lifetime.Time}, {lifetime.Memory}, ConstraintsMode.No, [{string.Join(",", dims)}], [{string.Join(",", strides)}], {false}),");
                bufferId++;
            }

            wr.WriteLine(@"]

source = {
    'name': [],
    'x': [],
    'y': [],
    'width': [],
    'height': [],
    'alpha': [],
    'color': [],
    'mem_interval': [],
    'time_interval': [],
    'shape': [],
    'stride': [],
}

y_range_max = 0
x_range_max = 0
color_dict = {}
for buffer in buffers:
  source['name'].append(buffer.name)
  width = buffer.time_interval.end - buffer.time_interval.start
  x = buffer.time_interval.start + (width / 2)
  height = buffer.mem_interval.end - buffer.mem_interval.start
  y = buffer.mem_interval.start + (height / 2)
  y_range_max = max(y_range_max, y)
  x_range_max = max(x_range_max, buffer.time_interval.end)
  source['x'].append(x)
  source['y'].append(y)
  source['width'].append(width)
  source['height'].append(height)
  color = color_dict.get(buffer.name)
  if color == None:
    color = next(colors)
    color_dict[buffer.name] = color
  source['color'].append(color)
  source['alpha'].append(0.2 if buffer.inplace else 1.0)
  source['time_interval'].append(str(buffer.time_interval))
  source['mem_interval'].append(str(buffer.mem_interval))
  source['shape'].append(','.join([str(s) for s in buffer.shape]))
  source['stride'].append(','.join([str(s) for s in buffer.stride]))

source = ColumnDataSource(source)
hover = HoverTool(tooltips=[('name', '@name'), ('time_interval', '@time_interval'), ('mem_interval', '@mem_interval'),
                            ('shape', '@shape'), ('stride', '@stride')])

p = figure(tools=[hover, WheelPanTool(), SaveTool(), WheelZoomTool(), ResetTool()], width=1280, height=720,
           y_range=(0, y_range_max * 1.2), x_range=(-1, x_range_max + 1),
           title='Local Buffer LifeTime (by Steps)')
p.rect(x='x', y='y', width='width', height='height', fill_color='color', legend_field='name', fill_alpha='alpha', source=source)
p.xaxis.axis_label = 'Time (steps)'
p.outline_line_color = None");

            wr.WriteLine($@"
save(p, filename='{group.Key}.html')
show(p)");
        }
    }

    private sealed class BufferReplacer : ExprRewriter
    {
        private readonly IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> _buffers;

        public BufferReplacer(IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> buffers)
        {
            _buffers = buffers;
        }

        protected override BaseExpr RewriteLeafPhysicalBuffer(TIR.PhysicalBuffer expr)
        {
            if (_buffers.TryGetValue(expr, out var lifetime))
            {
                var start = Tensor.FromScalar((ulong)lifetime.Memory.Start);
                return expr.With(start: start);
            }

            return expr;
        }
    }

    private sealed record ReadOnlyDataKey(string ElementType, string Dimensions, string Strides, int ByteLength, string Hash);
}
