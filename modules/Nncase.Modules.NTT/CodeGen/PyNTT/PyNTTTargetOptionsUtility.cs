// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Targets;

namespace Nncase.CodeGen.PyNTT;

internal static class PyNTTTargetOptionsUtility
{
    public static PyNTTTargetOptions Get(CompileOptions compileOptions)
    {
        if (compileOptions.TargetOptions is PyNTTTargetOptions pynttOptions)
        {
            return pynttOptions;
        }

        throw new InvalidOperationException("PyNTT codegen requires PyNTTTargetOptions.");
    }
}
