// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.TIR;

namespace Nncase.CodeGen.PyNTT;

internal static class PyNTTPrimFunctionRoles
{
    public static bool IsAutoTilingDeviceFunction(PrimFunction function)
        => function.Name.StartsWith("device_func", StringComparison.Ordinal);
}
