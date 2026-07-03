// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.CodeGen.PyNTT;

internal sealed class PyNTTLinkableFunction : ILinkableFunction
{
    public PyNTTLinkableFunction(uint id, BaseFunction sourceFunction, PyNTTGeneratedKernelSource generatedKernelSource, PyNTTRDataBundle rdataBundle)
    {
        Id = id;
        SourceFunction = sourceFunction;
        GeneratedKernelSource = generatedKernelSource;
        RDataBundle = rdataBundle;
        Text = new MemoryStream();
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public PyNTTGeneratedKernelSource GeneratedKernelSource { get; }

    public PyNTTRDataBundle RDataBundle { get; }

    public IEnumerable<FunctionRef> FunctionRefs => Enumerable.Empty<FunctionRef>();

    public Stream Text { get; }

    public IReadOnlyList<ILinkedSection> Sections => Array.Empty<ILinkedSection>();
}

internal sealed record PyNTTRDataBundle(
    string RData,
    long RDataBytes,
    string ChipLocalRData,
    long ChipLocalRDataBytes,
    string[] BlockLocalRDatas,
    long BlockLocalRDataBytes)
{
    public static PyNTTRDataBundle Empty { get; } = new(
        string.Empty,
        0,
        string.Empty,
        0,
        Array.Empty<string>(),
        0);
}
