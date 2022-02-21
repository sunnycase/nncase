﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

internal class MatchProvider : IMatchProvider
{
    public IMatchResult? Match(Expr expr, Pattern pattern)
    {
        return Matcher.Match(pattern, expr);
    }
}
