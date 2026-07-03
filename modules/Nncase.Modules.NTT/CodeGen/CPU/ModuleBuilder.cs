// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Targets;

namespace Nncase.CodeGen.NTT;

/// <summary>
/// K230CoreModule builder.
/// </summary>
public sealed class NTTModuleBuilder : IModuleBuilder
{
    private readonly SectionManager _sectionManager;
    private readonly BinaryWriter _rdataWriter;
    private readonly BinaryWriter[] _blockLocalRdataWriters;

    public NTTModuleBuilder(string moduleKind, CompileOptions options)
    {
        var targetOptions = (NTTTargetOptions)options.TargetOptions;
        var hierarchies = targetOptions.Hierarchies[0];
        ModuleKind = moduleKind;
        _sectionManager = new();
        _rdataWriter = _sectionManager.GetWriter(WellknownSectionNames.Rdata);

        var shardCount = TensorUtilities.GetProduct(hierarchies);
        var blocksCount = shardCount;
        _blockLocalRdataWriters = new BinaryWriter[blocksCount];
        for (int i = 0; i < _blockLocalRdataWriters.Length; i++)
        {
            _blockLocalRdataWriters[i] = _sectionManager.GetWriter(WellknownSectionNames.BlockLocalRdata, i);
        }

        CompileOptions = options;
    }

    public CompileOptions CompileOptions { get; }

    /// <inheritdoc/>
    public string ModuleKind { get; }

    /// <inheritdoc/>
    public ILinkableModule Build(IReadOnlyList<BaseFunction> functions)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;

        // 1. write the module header
        using (var writer = _sectionManager.GetWriter(LinkedModule.ModuleHeaderSectionName))
        {
            var header = default(ModuleDescHeader);
            var placement = new Placement(targetOptions.Hierarchies[0], targetOptions.HierarchyNames, targetOptions.HierarchyLevels);
            header.BlockDim = (uint)Math.Max(1, placement.GetPhysicalLevelSize('b'));
            header.ChipDim = (uint)Math.Max(1, placement.GetPhysicalLevelSize('c') * placement.GetPhysicalLevelSize('d'));

            writer.Write(ref header);
        }

        var linkableFunctions = functions.OfType<BaseFunction>().Select((f, i) => new FunctionBuilder((uint)i, _rdataWriter, _blockLocalRdataWriters, (Targets.NTTTargetOptions)CompileOptions.TargetOptions).Build(f)).ToArray();
        _rdataWriter.Flush();
        var blockLocalRdataContents = Enumerable.Range(0, _blockLocalRdataWriters.Length).Select(i =>
        {
            _blockLocalRdataWriters[i].Flush();
            return _sectionManager.GetContent(WellknownSectionNames.BlockLocalRdata, i)!;
        }).ToArray();

        return new LinkableModule(ModuleKind, _sectionManager.GetContent(LinkedModule.ModuleHeaderSectionName)!, _sectionManager.GetContent(WellknownSectionNames.Rdata)!, blockLocalRdataContents, linkableFunctions, CompileOptions);
    }
}
