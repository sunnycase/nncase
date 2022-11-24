﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen.StackVM;

internal class StackVMLinkableModule : LinkableModule
{
    public HashSet<ModuleType> CustomCallModules;

    public StackVMLinkableModule(IReadOnlyList<ILinkableFunction> functions, SectionManager sectionManager)
        : base(functions, sectionManager)
    {
        CustomCallModules = new();
        foreach (var funcs in functions.OfType<StackVMLinkableFunction>())
        {
            CustomCallModules.UnionWith(funcs.CustomCallModules);
        }

        var writer = SectionManager.GetWriter(".custom_calls");
        writer.Write((uint)CustomCallModules.Count);
        foreach (var item in CustomCallModules)
        {
            writer.Write(item.Types.AsSpan());
        }
    }

    protected override ILinkedModule CreateLinkedModule(IReadOnlyList<LinkedFunction> linkedFunctions, byte[] text)
    {
        return new StackVMLinkedModule(linkedFunctions, text,
          SectionManager.GetContent(WellknownSectionNames.Rdata),
          SectionManager.GetContent(".custom_calls"));
    }
}
