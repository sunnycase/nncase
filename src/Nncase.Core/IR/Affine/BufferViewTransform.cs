// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.Affine;

/// <summary>
/// Describes two tensor index spaces over one common affine domain.
/// </summary>
public sealed record BufferViewTransform
{
    /// <summary>
    /// Initializes a new instance of the <see cref="BufferViewTransform"/> class.
    /// </summary>
    public BufferViewTransform(AffineMap sourceMap, AffineMap resultMap, IRArray<Dimension> domainBounds)
    {
        if (sourceMap.Domains.Length != resultMap.Domains.Length || sourceMap.Domains.Length != domainBounds.Count)
        {
            throw new ArgumentException(
                $"Buffer view domain mismatch: source={sourceMap.Domains.Length}, result={resultMap.Domains.Length}, bounds={domainBounds.Count}.");
        }

        if (sourceMap.Symbols.Length != 0 || resultMap.Symbols.Length != 0)
        {
            throw new NotSupportedException("Buffer views do not support symbolic affine-map parameters.");
        }

        SourceMap = sourceMap;
        ResultMap = resultMap;
        DomainBounds = domainBounds;
    }

    /// <summary>
    /// Gets the common-domain to source-index map.
    /// </summary>
    public AffineMap SourceMap { get; }

    /// <summary>
    /// Gets the common-domain to result-index map.
    /// </summary>
    public AffineMap ResultMap { get; }

    /// <summary>
    /// Gets the exact logical bounds of the common domain.
    /// </summary>
    public IRArray<Dimension> DomainBounds { get; }

    /// <summary>
    /// Creates an identity storage view for a ranked tensor shape.
    /// </summary>
    public static BufferViewTransform Identity(Shape shape)
    {
        if (shape is not RankedShape rankedShape)
        {
            throw new ArgumentException("Buffer views require a ranked shape.", nameof(shape));
        }

        var map = AffineMap.Identity(rankedShape.Rank);
        return new BufferViewTransform(map, map, new IRArray<Dimension>(rankedShape.Dimensions));
    }

    /// <inheritdoc/>
    public override string ToString() => $"domain=[{string.Join(", ", DomainBounds)}], source={SourceMap}, result={ResultMap}";
}
