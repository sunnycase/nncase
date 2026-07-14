// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;
using Nncase.Schedule.MonteCarloTreeSearch;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Searches fusion and lexical loop order. Structural decisions form an
/// ordered tree, while every rollout is costed by the exact tile and placement
/// solver used for lowering.
/// </summary>
internal sealed class TileStructuralSearch
{
    private readonly ImmutableArray<StructuralDecision> _decisions;
    private readonly Func<TileStructuralSchedule, (TileExecutionPlan? Plan, string Failure)> _evaluate;
    private readonly Dictionary<TileStructuralSchedule, (TileExecutionPlan? Plan, string Failure)> _evaluationMemo = new();
    private readonly int _levelCount;
    private readonly TileRegion _region;
    private TileExecutionPlan? _best;

    public TileStructuralSearch(
        TileRegion region,
        int levelCount,
        Func<TileStructuralSchedule, (TileExecutionPlan? Plan, string Failure)> evaluate)
    {
        _region = region;
        _levelCount = levelCount;
        _evaluate = evaluate;
        _decisions = BuildDecisions(region);
    }

    public TileExecutionPlan Run()
    {
        var initialSchedule = TileStructuralSchedule.Create(_region);
        foreach (var use in _region.Uses.Where(use => use.RequiredMemoryScope == MemoryAccessScope.Chip))
        {
            initialSchedule = initialSchedule.WithFusionLevel(use.Id, _levelCount - 1);
        }

        var initial = new SearchState(
            this,
            initialSchedule,
            0,
            "root");
        var root = new StructuralSearchNode(initial);

        Evaluate(initial.Schedule);
        root.Update(initial.RollOut());
        new StructuralSearcher(
            GetSearchIterations(),
            () => Math.Max(1D, _best?.ObjectiveValue ?? 1D)).Search(root);

        DumpSearch(root);
        return _best ?? throw new SolveFailedException(
            $"Hierarchical AutoTiling found no feasible structural schedule. " +
            $"Evaluated {_evaluationMemo.Count} terminal schedules: " +
            string.Join(
                " | ",
                _evaluationMemo.Values
                    .Where(value => value.Plan is null)
                    .Select(value => value.Failure)
                    .Distinct()
                    .Take(4)));
    }

    private static ImmutableArray<StructuralDecision> BuildDecisions(TileRegion region)
    {
        var fusionDecisions = region.Uses
            .Where(use => use.RequiredMemoryScope != MemoryAccessScope.Chip)
            .OrderByDescending(use => use.MaximumBytes)
            .ThenBy(use => use.Id.ProducerOpId)
            .ThenBy(use => use.Id.ProducerOutputIndex)
            .ThenBy(use => use.Id.ConsumerOpId)
            .ThenBy(use => use.Id.ConsumerAccessIndex)
            .Select(use => (StructuralDecision)new FusionDecision(use.Id))
            .ToArray();
        var loopDecisions = region.Scopes
            .OrderByDescending(scope => scope.Rank)
            .ThenBy(scope => scope.Id.AnchorOpId)
            .ThenBy(scope => scope.Id.Level)
            .SelectMany(scope => Enumerable.Range(0, Math.Max(0, scope.Rank - 1))
                .Select(position => (StructuralDecision)new LoopAxisDecision(scope.Id, position)))
            .ToArray();

        // Interleave dataflow and loop decisions so a bounded search can reach
        // both dimensions of the structural schedule.
        var decisions = ImmutableArray.CreateBuilder<StructuralDecision>(fusionDecisions.Length + loopDecisions.Length);
        for (int index = 0; index < Math.Max(fusionDecisions.Length, loopDecisions.Length); index++)
        {
            if (index < fusionDecisions.Length)
            {
                decisions.Add(fusionDecisions[index]);
            }

            if (index < loopDecisions.Length)
            {
                decisions.Add(loopDecisions[index]);
            }
        }

        return decisions.MoveToImmutable();
    }

    private static ImmutableArray<int> MoveAxisToPosition(
        ImmutableArray<int> order,
        int axis,
        int position)
    {
        var sourcePosition = order.IndexOf(axis);
        if ((uint)position >= (uint)order.Length || sourcePosition < position)
        {
            throw new ArgumentException(
                $"Axis d{axis} cannot be selected at undecided loop position {position} in [{string.Join(", ", order)}].");
        }

        var builder = order.ToBuilder();
        builder.RemoveAt(sourcePosition);
        builder.Insert(position, axis);
        return builder.MoveToImmutable();
    }

    private static int GetSearchIterations()
    {
        const int defaultIterations = 0;
        var text = Environment.GetEnvironmentVariable("NNCASE_TILING_STRUCTURAL_SEARCH_STEPS");
        if (text is null)
        {
            return defaultIterations;
        }

        if (!int.TryParse(text, out var value) || value < 0)
        {
            throw new InvalidOperationException(
                $"NNCASE_TILING_STRUCTURAL_SEARCH_STEPS must be a non-negative integer, got '{text}'.");
        }

        return value;
    }

    private TileStructuralSchedule RollOut(SearchState state)
    {
        var schedule = state.Schedule;
        for (int index = state.NextDecision; index < _decisions.Length; index++)
        {
            if (_decisions[index] is not FusionDecision fusion)
            {
                continue;
            }

            // L0 is the innermost common execution scope. A rollout must try
            // the tightest legal optional fusion first; chip-visible phase
            // composition is already fixed in the initial schedule.
            var use = _region.Uses.Single(candidate => candidate.Id == fusion.Use);
            foreach (var level in Enumerable.Range(0, _levelCount).Where(level => use.CanFuseAtLevel(level, _levelCount)))
            {
                var candidate = schedule.WithFusionLevel(fusion.Use, level);
                if (TileScheduleBuilder.TryBuild(_region, candidate, _levelCount, out _, out _))
                {
                    schedule = candidate;
                    break;
                }
            }
        }

        return schedule;
    }

    private SearchState? Apply(SearchState state, StructuralAction action)
    {
        if (state.NextDecision >= _decisions.Length)
        {
            throw new InvalidOperationException("Cannot apply an action to a terminal structural schedule.");
        }

        var decision = _decisions[state.NextDecision];
        var schedule = (decision, action) switch
        {
            (FusionDecision expected, FusionAction selected) when expected.Use == selected.Use =>
                state.Schedule.WithFusionLevel(selected.Use, selected.Level),
            (LoopAxisDecision expected, LoopAxisAction selected)
                when expected.Scope == selected.Scope && expected.Position == selected.Position =>
                state.Schedule.WithLoopOrder(
                    selected.Scope,
                    MoveAxisToPosition(
                        state.Schedule.GetLoopOrder(selected.Scope),
                        selected.Axis,
                        selected.Position)),
            _ => throw new InvalidOperationException(
                $"Structural action {action} does not implement decision {decision}."),
        };
        if (!TileScheduleBuilder.TryBuild(_region, schedule, _levelCount, out _, out _))
        {
            return null;
        }

        return new SearchState(
            this,
            schedule,
            state.NextDecision + 1,
            $"{state.SearchPath()}/{action}");
    }

    private ImmutableArray<StructuralAction> GetActions(SearchState state)
    {
        if (state.NextDecision >= _decisions.Length)
        {
            return ImmutableArray<StructuralAction>.Empty;
        }

        return _decisions[state.NextDecision] switch
        {
            FusionDecision fusion => Enumerable.Range(0, _levelCount)
                .Where(level => _region.Uses.Single(use => use.Id == fusion.Use).CanFuseAtLevel(level, _levelCount))
                .Append(-1)
                .Select(level => (StructuralAction)new FusionAction(fusion.Use, level))
                .ToImmutableArray(),
            LoopAxisDecision loop => state.Schedule.GetLoopOrder(loop.Scope)
                .Skip(loop.Position)
                .Select(axis => (StructuralAction)new LoopAxisAction(loop.Scope, loop.Position, axis))
                .ToImmutableArray(),
            _ => throw new InvalidOperationException(
                $"Unsupported structural decision {_decisions[state.NextDecision].GetType().Name}."),
        };
    }

    private (TileExecutionPlan? Plan, string Failure) Evaluate(TileStructuralSchedule schedule)
    {
        if (!_evaluationMemo.TryGetValue(schedule, out var evaluation))
        {
            evaluation = _evaluate(schedule);
            _evaluationMemo.Add(schedule, evaluation);
        }

        if (evaluation.Plan is { } plan && IsBetter(plan, _best))
        {
            _best = plan;
        }

        return evaluation;

        bool IsBetter(TileExecutionPlan candidate, TileExecutionPlan? incumbent)
        {
            if (incumbent is null || candidate.ObjectiveValue != incumbent.ObjectiveValue)
            {
                return incumbent is null || candidate.ObjectiveValue < incumbent.ObjectiveValue;
            }

            var candidateFusionCount = CountFusions(candidate);
            var incumbentFusionCount = CountFusions(incumbent);
            if (candidateFusionCount != incumbentFusionCount)
            {
                return candidateFusionCount > incumbentFusionCount;
            }

            return GetFusionTightness(candidate) > GetFusionTightness(incumbent);
        }

        static int CountFusions(TileExecutionPlan plan)
            => plan.Structure.FusionLevels.Count(item => item.Value >= 0);

        int GetFusionTightness(TileExecutionPlan plan)
            => plan.Structure.FusionLevels
                .Where(item => item.Value >= 0)
                .Sum(item => _levelCount - item.Value);
    }

    private void DumpSearch(StructuralSearchNode root)
    {
        if (!Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            return;
        }

        using var stream = Diagnostics.DumpScope.Current.OpenFile("tile_structural_search.yaml");
        using var writer = new StreamWriter(stream);
        writer.WriteLine($"decision_count: {_decisions.Length}");
        writer.WriteLine("decisions:");
        foreach (var (decision, index) in _decisions.Select((decision, index) => (decision, index)))
        {
            writer.WriteLine($"  - index: {index}");
            writer.WriteLine($"    decision: {decision}");
        }

        writer.WriteLine("tree:");
        root.Dump(writer, 1);
    }

    private abstract record StructuralDecision;

    private sealed record FusionDecision(TileUseId Use) : StructuralDecision
    {
        public override string ToString() => $"fusion({Use})";
    }

    private sealed record LoopAxisDecision(TileScopeId Scope, int Position) : StructuralDecision
    {
        public override string ToString() => $"loop-axis({Scope}, position={Position})";
    }

    private abstract record StructuralAction;

    private sealed record FusionAction(TileUseId Use, int Level) : StructuralAction
    {
        public override string ToString() => $"fusion({Use}, L{Level})";
    }

    private sealed record LoopAxisAction(TileScopeId Scope, int Position, int Axis) : StructuralAction
    {
        public override string ToString() => $"loop-axis({Scope}, position={Position}, d{Axis})";
    }

    private sealed class SearchState : IEnvironmentState<StructuralAction>
    {
        private readonly List<StructuralAction> _remainingActions;
        private readonly TileStructuralSearch _owner;
        private readonly string _searchPath;
        private bool _isEvaluated;

        public SearchState(
            TileStructuralSearch owner,
            TileStructuralSchedule schedule,
            int nextDecision,
            string searchPath)
        {
            _owner = owner;
            Schedule = schedule;
            NextDecision = nextDecision;
            _searchPath = searchPath;
            _remainingActions = owner.GetActions(this).ToList();
            ObjectValue = long.MaxValue;
        }

        public long ObjectValue { get; private set; }

        public TileStructuralSchedule Schedule { get; }

        public int NextDecision { get; }

        public string SearchPath() => _searchPath;

        public int LegalActions() => _remainingActions.Count;

        public StructuralAction GetNextAction(int index)
        {
            if ((uint)index >= (uint)_remainingActions.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index), index, "Structural action index is out of range.");
            }

            var action = _remainingActions[index];
            _remainingActions.RemoveAt(index);
            return action;
        }

        public IEnvironmentState<StructuralAction>? PerformAction(StructuralAction action)
            => _owner.Apply(this, action);

        public double RollOut()
        {
            if (!_isEvaluated)
            {
                var (plan, failure) = _owner.Evaluate(_owner.RollOut(this));
                ObjectValue = plan?.ObjectiveValue ?? long.MaxValue;
                _isEvaluated = true;
            }

            return ObjectValue;
        }
    }

    private sealed class StructuralSearchNode : SearchNode<StructuralAction>
    {
        private const double Exploration = 1.4142135623730951;

        public StructuralSearchNode(SearchState state)
            : base(state)
        {
            QualityValue = double.PositiveInfinity;
        }

        public StructuralSearchNode(
            SearchNode<StructuralAction> parent,
            SearchState state,
            StructuralAction action)
            : base(parent, state)
        {
            Action = action;
            QualityValue = double.PositiveInfinity;
        }

        public StructuralAction? Action { get; }

        private SearchState StructuralState => (SearchState)State;

        public double Score(double normalization)
        {
            if (VisitTimes == 0)
            {
                return double.PositiveInfinity;
            }

            var quality = double.IsPositiveInfinity(QualityValue)
                ? -double.MaxValue
                : -(QualityValue / normalization);
            var parentVisits = Math.Max(1, Parent?.VisitTimes ?? 1);
            return quality + (Exploration * Math.Sqrt(Math.Log(parentVisits + 1D) / VisitTimes));
        }

        public override void Update(double reward)
        {
            VisitTimes++;
            QualityValue = Math.Min(QualityValue, reward);
            Parent?.Update(reward);
        }

        public override void Dump(System.CodeDom.Compiler.IndentedTextWriter writer)
            => throw new NotSupportedException("Structural search uses its YAML dump overload.");

        public void Dump(StreamWriter writer, int depth)
        {
            var indent = new string(' ', depth * 2);
            writer.WriteLine($"{indent}- action: {Action?.ToString() ?? "root"}");
            writer.WriteLine($"{indent}  next_decision: {StructuralState.NextDecision}");
            writer.WriteLine($"{indent}  visits: {VisitTimes}");
            writer.WriteLine($"{indent}  best_objective: {QualityValue}");
            writer.WriteLine($"{indent}  schedule: {StructuralState.Schedule}");
            writer.WriteLine($"{indent}  children:");
            foreach (var child in Children.Cast<StructuralSearchNode>())
            {
                child.Dump(writer, depth + 2);
            }
        }
    }

    private sealed class StructuralSearcher : Searcher<StructuralAction>
    {
        private readonly Func<double> _normalization;

        public StructuralSearcher(int searchTimes, Func<double> normalization)
            : base(searchTimes)
        {
            _normalization = normalization;
        }

        public override bool Selection(
            SearchNode<StructuralAction> node,
            out SearchNode<StructuralAction> selected)
        {
            while (node.State.LegalActions() == 0 && node.Children.Count != 0)
            {
                node = node.Children
                    .Cast<StructuralSearchNode>()
                    .OrderByDescending(child => child.Score(_normalization()))
                    .ThenBy(child => child.Action?.ToString(), StringComparer.Ordinal)
                    .First();
            }

            selected = node;
            return true;
        }

        public override SearchNode<StructuralAction>? Expand(SearchNode<StructuralAction> node)
        {
            while (node.State.LegalActions() != 0)
            {
                var action = node.State.GetNextAction(0);
                if (node.State.PerformAction(action) is SearchState state)
                {
                    return new StructuralSearchNode(node, state, action);
                }
            }

            return null;
        }

        public override double Simulation(SearchNode<StructuralAction> node)
            => node.State.RollOut();

        public override void BackPropagate(SearchNode<StructuralAction> node, double reward)
            => node.Update(reward);
    }
}
