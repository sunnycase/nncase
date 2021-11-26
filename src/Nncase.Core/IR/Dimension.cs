﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Dimension kind.
    /// </summary>
    public enum DimensionKind
    {
        /// <summary>
        /// Unknown dimension.
        /// </summary>
        Unknown,

        /// <summary>
        /// Fixed dimesnion.
        /// </summary>
        Fixed,
    }

    /// <summary>
    /// Shape dimension.
    /// </summary>
    public struct Dimension : IEquatable<Dimension>
    {
        /// <summary>
        /// An unknown dimension.
        /// </summary>
        public static readonly Dimension Unknown = default;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dimension"/> struct.
        /// </summary>
        /// <param name="value">Dimension value.</param>
        public Dimension(int value)
        {
            if (value == 0)
            {
                throw new InvalidOperationException("Dimension should not be 0");
            }
            Kind = DimensionKind.Fixed;
            Value = value;
        }

        /// <summary>
        /// Gets kind.
        /// </summary>
        public DimensionKind Kind { get; }

        /// <summary>
        /// Gets value.
        /// </summary>
        public int? Value { get; }

        /// <summary>
        /// Gets FixedValue.
        /// </summary>
        public int FixedValue
        {
            get => Value ??
               throw new InvalidOperationException("Only Can Get It When Shape Is Fixed !");
        }

        /// <summary>
        /// Gets a value indicating whether unknown.
        /// </summary>
        public bool IsUnknown => Kind == DimensionKind.Unknown;

        /// <summary>
        /// Gets a value indicating whether fixed.
        /// </summary>
        public bool IsFixed => Kind == DimensionKind.Fixed;

        /// <summary>
        /// Convert <see cref="long"/> to a fixed <see cref="Dimension"/>.
        /// </summary>
        /// <param name="value">Dimension value.</param>
        public static implicit operator Dimension(int value) => new(value);

        public static bool operator ==(Dimension left, Dimension right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Dimension left, Dimension right)
        {
            return !(left == right);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return Value?.ToString() ?? "?";
        }

        public override bool Equals(object? obj)
        {
            return obj is Dimension dimension && Equals(dimension);
        }

        public bool Equals(Dimension other)
        {
            return Kind == other.Kind &&
                   Value == other.Value;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Kind, Value);
        }

        public static Dimension operator +(Dimension lhs, Dimension rhs) => (lhs.IsFixed, lhs.IsFixed) switch
        {
            (true, true) => lhs.FixedValue + rhs.FixedValue,
            (_, _) => Dimension.Unknown
        };

        public static Dimension operator -(Dimension lhs, Dimension rhs) => (lhs.IsFixed, lhs.IsFixed) switch
        {
            (true, true) => lhs.FixedValue - rhs.FixedValue,
            (_, _) => Dimension.Unknown
        };

        public static Dimension operator *(Dimension lhs, Dimension rhs) => (lhs.IsFixed, lhs.IsFixed) switch
        {
            (true, true) => lhs.FixedValue * rhs.FixedValue,
            (_, _) => Dimension.Unknown
        };

        public static Dimension operator /(Dimension lhs, Dimension rhs) => (lhs.IsFixed, lhs.IsFixed) switch
        {
            (true, true) => lhs.FixedValue / rhs.FixedValue,
            (_, _) => Dimension.Unknown
        };
    }
}
