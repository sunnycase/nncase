// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Utilities;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule;

public static class TilingUtilities
{
    public static Shape GetBufferShape(Expr buffer, bool maxShape)
    {
        return buffer.CheckedType switch
        {
            TensorType t => maxShape ? CompilerServices.GetMaxShape(t.Shape) : t.Shape,
            DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt, maxShape ? DistributedUtility.DivideFlags.MaxShape | DistributedUtility.DivideFlags.FloorDiv : DistributedUtility.DivideFlags.FloorDiv).Shape,
            _ => throw new NotSupportedException(),
        };
    }

    /// <summary>
    /// some times we need to get the min/max value of an affine map.
    /// </summary>
    public static (Isl.multi_pw_aff MinMpa, Isl.multi_pw_aff MaxMpa) ToMinMaxMpa(AffineMap map)
    {
        var domains = string.Join(", ", Enumerable.Range(0, map.Domains.Length).Select(i => $"d{i}"));
        if (map.Symbols.Length > 0)
        {
            throw new NotSupportedException("Isl map does not support symbols yet.");
        }

        var minResults = StringUtility.Join(", ", map.Results.ToArray().Select(expr => expr.Offset switch
        {
            AffineConstant c => c.Value.ToString(),
            _ => expr.Offset.GetDisplayString(map.Symbols),
        }));

        var maxResults = StringUtility.Join(", ", map.Results.ToArray().Select(expr => expr.Offset switch
        {
            AffineConstant c => expr.Extent.GetDisplayString(map.Symbols),
            _ => expr.Offset.GetDisplayString(map.Symbols),
        }));

        return (new Isl.multi_pw_aff(Isl.ctx.Current, $"{{ [{domains}] -> [{minResults}] }}"),
                new Isl.multi_pw_aff(Isl.ctx.Current, $"{{ [{domains}] -> [{maxResults}] }}"));
    }
}
