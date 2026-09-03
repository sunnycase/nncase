// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;

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
        var primFunctions = functions.OfType<PrimFunction>().ToArray();
        var rdataPoolSize = primFunctions
            .SelectMany(function => function.SchedResult.Rdatas.Values)
            .Select(range => range.Max)
            .DefaultIfEmpty()
            .Max();
        var chipLocalRdataPoolSize = primFunctions
            .SelectMany(function => function.SchedResult.ChipLocalRdatas.Values)
            .Select(range => range.Max)
            .DefaultIfEmpty()
            .Max();
        var rdataAlignment = primFunctions
            .Select(function => function.SchedResult.RDataAlign)
            .DefaultIfEmpty(8UL)
            .Max();
        var chipLocalRdataAlignment = primFunctions
            .Select(function => function.SchedResult.ChipLocalRDataAlign)
            .DefaultIfEmpty(8UL)
            .Max();
        var chipLocalRdataBase = MathUtility.AlignUp(
            MathUtility.AlignUp(rdataPoolSize, rdataAlignment),
            chipLocalRdataAlignment);
        var mergedRdataPoolSize = checked(chipLocalRdataBase + chipLocalRdataPoolSize);

        // 1. write the module header
        using (var writer = _sectionManager.GetWriter(LinkedModule.ModuleHeaderSectionName))
        {
            var header = default(ModuleDescHeader);
            var placement = new Placement(targetOptions.Hierarchies[0], targetOptions.HierarchyNames, targetOptions.HierarchyLevels);
            header.BlockDim = (uint)Math.Max(1, placement.GetPhysicalLevelSize('b'));
            header.ChipDim = (uint)Math.Max(1, placement.GetPhysicalLevelSize('c') * placement.GetPhysicalLevelSize('d'));

            writer.Write(ref header);
        }

        var linkableFunctions = new List<ILinkableFunction>(functions.Count);
        uint publicFunctionId = 0;
        var hasPipelineWorker = primFunctions.Any(function => function.Role == FunctionRole.PipelineWorker);
        foreach (var function in functions)
        {
            var isPublic = function is PrimFunction primFunction &&
                (!hasPipelineWorker || primFunction.Role == FunctionRole.PipelineWorker);
            var functionId = isPublic ? publicFunctionId++ : 0;
            linkableFunctions.Add(new FunctionBuilder(
                functionId,
                _rdataWriter,
                _blockLocalRdataWriters,
                targetOptions,
                chipLocalRdataBase,
                mergedRdataPoolSize).Build(function));
        }
        _rdataWriter.Flush();
        var blockLocalRdataContents = Enumerable.Range(0, _blockLocalRdataWriters.Length).Select(i =>
        {
            _blockLocalRdataWriters[i].Flush();
            return _sectionManager.GetContent(WellknownSectionNames.BlockLocalRdata, i)!;
        }).ToArray();

        return new LinkableModule(ModuleKind, _sectionManager.GetContent(LinkedModule.ModuleHeaderSectionName)!, _sectionManager.GetContent(WellknownSectionNames.Rdata)!, blockLocalRdataContents, linkableFunctions, CompileOptions);
    }
}
