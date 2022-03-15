﻿using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;

namespace Nncase.SourceGenerator.Pattern;



internal class UsingComparer : IEqualityComparer<UsingDirectiveSyntax>
{
    public bool Equals(UsingDirectiveSyntax x, UsingDirectiveSyntax y)
    {
        return x.Name.GetFullName() == y.Name.GetFullName();
    }

    public int GetHashCode(UsingDirectiveSyntax obj)
    {
        return obj.Name.GetFullName().GetHashCode();
    }
}

[Generator]
public class PatternGenerator : ISourceGenerator
{
    public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new PatternReceiver());



    public void Execute(GeneratorExecutionContext context)
    {
        if (context.SyntaxContextReceiver is not PatternReceiver receiver)
            return;

        if (!receiver.Candidates.Any())
            return;

        var groupedCandidates = (from cand in receiver.Candidates
                                 group cand by cand.Op.ContainingNamespace into g
                                 select (g.Key, g.ToArray()));

        List<NamespaceDeclarationSyntax> namespaces = new();
        foreach (var (old_namespace, candidates) in groupedCandidates)
        {
            List<MemberDeclarationSyntax> members = new List<MemberDeclarationSyntax>();

            var pattern_name_params = new List<ParameterSyntax>()
                {
                    Parameter(Identifier("target_name")).WithType(ParseTypeName("string?")),
                    Parameter(Identifier("call_name")).WithType(ParseTypeName("string?")),
                };
            foreach (var cand in candidates)
            {
                // build the three pattern functional.
                foreach (var name_params in new List<List<ParameterSyntax?>>() { new(){ null,null },
                                                                         new(){ pattern_name_params[0],null },
                                                                    new(){ pattern_name_params[0],pattern_name_params[1] } })
                {
                    { // 1. build normal method
                      // 1.1 method params
                        var method_params = (from p in name_params
                                             where p is not null
                                             select p)
                                     .Concat(from p in cand.AttrParams
                                             select Parameter(Identifier(p.Name)).WithType(ParseTypeName(p.Type.ToDisplayString())))
                                     .Concat(from f in cand.ExprParams
                                             select Parameter(Identifier(f.Name.ToLower())).WithType(ParseTypeName("Pattern")));
                        var statements = new List<StatementSyntax>();
                        {
                            // 1.2 build condition
                            var condition = string.Join("&&", (from p in cand.AttrParams select $"(x.{p.Name} == {p.Name})").DefaultIfEmpty("true"));
                            var inputs = string.Join(", ", from f in cand.ExprParams select f.Name.ToLower());
                            // 1.3 build method return
                            //var x = name_params[0];
                            statements.Add(ParseStatement(@$"return new(
new OpPattern<{cand.Op.ToDisplayString()}>(x => {condition}, {(name_params[0] != null ? "target_name" : "null")}), 
new VArgsPattern (new[]{{ {inputs} }}, null),
{(name_params[1] != null ? "call_name" : "null")});"));
                        }
                        // 1.4. build method body
                        var method = MethodDeclaration(ParseTypeName("CallPattern"), "Is" + cand.Op.Name)
                                     .WithParameterList(ParameterList(SeparatedList(method_params)))
                                     .WithModifiers(TokenList(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.StaticKeyword)))
                                    .WithBody(Block(statements));
                        members.Add(method);
                    }
                    { // 2. build funciton with condition
                      // 2.1 method params
                        var method_params = (from p in name_params
                                             where p is not null
                                             select p)
                                     .Concat(new[] {Parameter(Identifier("condition"))
                                             .WithType(ParseTypeName($"Func<{cand.Op.ToDisplayString()},bool>"))})
                                     .Concat(from f in cand.ExprParams
                                             select Parameter(Identifier(f.Name.ToLower()))
                                             .WithType(ParseTypeName("Pattern")));
                        var statements = new List<StatementSyntax>();
                        {
                            // 1.2 build condition
                            var inputs = string.Join(", ", from f in cand.ExprParams select f.Name.ToLower());
                            // 1.3 build method return
                            statements.Add(ParseStatement(@$"return new(
new OpPattern<{cand.Op.ToDisplayString()}>(condition, {(name_params[0] != null ? "target_name" : "null")}),
new VArgsPattern( new [] {{ {inputs} }}, null ),
{(name_params[1] != null ? "call_name" : "null")});"));
                        }
                        // 1.4. build method body
                        var method = MethodDeclaration(ParseTypeName("CallPattern"), "Is" + cand.Op.Name)
                                     .WithParameterList(ParameterList(SeparatedList(method_params)))
                                     .WithModifiers(TokenList(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.StaticKeyword)))
                                    .WithBody(Block(statements));
                        members.Add(method);
                    }
                }
            }
            // 4. build static class
            List<ClassDeclarationSyntax> classes = new();
            {
                var class_name = old_namespace.MetadataName.Split('.').Last();

                var @class = ClassDeclaration(Identifier(class_name))
                               .WithModifiers(TokenList(
                                   Token(SyntaxKind.PublicKeyword),
                                   Token(SyntaxKind.StaticKeyword),
                                   Token(SyntaxKind.PartialKeyword)))
                               .AddMembers(members.ToArray());
                classes.Add(@class);
            }
            //5. build namespace
            var arr = old_namespace.ToDisplayString().Split('.');
            arr[arr.Length - 1] = "F";
            var @namespcae = NamespaceDeclaration(ParseName(string.Join(".", arr).Replace("IR", "PatternMatch")))
                .AddMembers(classes.ToArray());
            namespaces.Add(namespcae);
        }
        var compilationUnit = CompilationUnit().
                AddMembers(namespaces.ToArray()).
                //AddUsings(usings.ToArray()).
                NormalizeWhitespace();
        context.AddSource("Ops.Pattern", SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText());
    }
}
