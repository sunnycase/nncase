// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen.PyNTT;

internal static class PyNTTPrimFunctionRoles
{
    public static bool IsDispatchFunction(PrimFunction function)
        => function.Role == FunctionRole.Dispatch;

    public static bool IsScheduledRegionFunction(PrimFunction function)
        => function.Role == FunctionRole.ScheduledRegion;
}
