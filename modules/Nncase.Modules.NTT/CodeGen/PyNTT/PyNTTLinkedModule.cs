// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.PyNTT;

internal sealed class PyNTTLinkedModule : ILinkedModule
{
    public PyNTTLinkedModule(string moduleKind, IReadOnlyList<ILinkedFunction> functions, IReadOnlyList<ILinkedSection> sections)
    {
        ModuleKind = moduleKind;
        Functions = functions;
        Sections = sections;
    }

    public string ModuleKind { get; }

    public uint Version => 0;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
