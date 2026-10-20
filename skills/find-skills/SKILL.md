---
name: find-skills
description: Helps users discover and install agent skills when they ask questions like "how do I do X", "find a skill for X", "is there a skill that can...", or express interest in extending capabilities. This skill should be used when the user is looking for functionality that might exist as an installable skill.
---

# Find Skills

This skill helps you discover and install skills from the open agent skills ecosystem.

## Where Skills Are Installed

This agent loads its skills from the `skills/` folder in its working directory
(one subfolder per skill, each containing a `SKILL.md`). Skills must be
installed there, and nowhere else:

- Always install at project level (in the working directory). Never use `-g` / `--global`.
- Always pass `--agent openclaw --copy`. The Skills CLI has no option for a custom
  folder, but the `openclaw` agent's project folder is `skills/`, so this puts the
  skill in `skills/<skill-name>/`. It does not install anything for OpenClaw or any
  other coding agent, so do not "correct" this to another agent name.
- `--copy` writes real files instead of symlinks, which need extra permissions on Windows.
- Shell commands get no keyboard input, so always use `npx --yes` and the CLI's `-y`
  flag, or the command will hang waiting for a confirmation.

## When to Use This Skill

Use this skill when the user:

- Asks "how do I do X" where X might be a common task with an existing skill
- Says "find a skill for X" or "is there a skill for X"
- Asks "can you do X" where X is a specialized capability
- Expresses interest in extending agent capabilities
- Wants to search for tools, templates, or workflows
- Mentions they wish they had help with a specific domain (design, testing, deployment, etc.)

## What is the Skills CLI?

The Skills CLI (`npx skills`) is the package manager for the open agent skills ecosystem. Skills are modular packages that extend agent capabilities with specialized knowledge, workflows, and tools.

**Key commands:**

- `npx --yes skills find [query] [--owner <owner>]` - Search for skills by keyword, optionally scoped to a GitHub owner
- `npx --yes skills add <owner/repo> --skill <skill-name> --agent openclaw --copy -y` - Install one skill into `skills/<skill-name>/`
- `npx --yes skills add <owner/repo> --list` - List the skills in a repository without installing anything

**Browse skills at:** https://skills.sh/

## How to Help Users Find Skills

### Step 1: Understand What They Need

When a user asks for help with something, identify:

1. The domain (e.g., React, testing, design, deployment)
2. The specific task (e.g., writing tests, creating animations, reviewing PRs)
3. Whether this is a common enough task that a skill likely exists

### Step 2: Check the Leaderboard First

Before running a CLI search, check the [skills.sh leaderboard](https://skills.sh/) to see if a well-known skill already exists for the domain. The leaderboard ranks skills by total installs, surfacing the most popular and battle-tested options.

For example, top skills for web development include:
- `vercel-labs/agent-skills` — React, Next.js, web design (100K+ installs each)
- `anthropics/skills` — Frontend design, document processing (100K+ installs)

### Step 3: Search for Skills

If the leaderboard doesn't cover the user's need, run the find command:

```bash
npx --yes skills find [query] [--owner <owner>]
```

For example:

- User asks "how do I make my React app faster?" → `npx --yes skills find react performance`
- User asks "can you help me with PR reviews?" → `npx --yes skills find pr review`
- User asks "I need to create a changelog" → `npx --yes skills find changelog`

Search results are listed as `<owner/repo>@<skill-name>`.

### Step 4: Verify Quality Before Recommending

**Do not recommend a skill based solely on search results.** Always verify:

1. **Install count** — Prefer skills with 1K+ installs. Be cautious with anything under 100.
2. **Source reputation** — Official sources (`vercel-labs`, `anthropics`, `microsoft`) are more trustworthy than unknown authors.
3. **GitHub stars** — Check the source repository. A skill from a repo with <100 stars should be treated with skepticism.

### Step 5: Present Options to the User

When you find relevant skills, present them to the user with:

1. The skill name and what it does
2. The install count and source
3. A link to learn more at skills.sh

Example response:

```
I found a skill that might help! The "react-best-practices" skill provides
React and Next.js performance optimization guidelines from Vercel Engineering.
(185K installs)

Learn more: https://skills.sh/vercel-labs/agent-skills/react-best-practices
```

### Step 6: Offer to Install

Only install a skill when the user has asked you to. Split the search result
`<owner/repo>@<skill-name>` into its two parts and run, from the working directory:

```bash
npx --yes skills add <owner/repo> --skill <skill-name> --agent openclaw --copy -y
```

For example, for `vercel-labs/agent-skills@react-best-practices`:

```bash
npx --yes skills add vercel-labs/agent-skills --skill react-best-practices --agent openclaw --copy -y
```

Then confirm that `skills/<skill-name>/SKILL.md` exists. The CLI also records the
install in `skills-lock.json`; leave that file alone.

Newly installed skills are only added to your skill list the next time the agent
starts. To use one in the current conversation, read
`skills/<skill-name>/SKILL.md` and follow its instructions. Tell the user it will be
listed automatically from their next session.

## Common Skill Categories

When searching, consider these common categories:

| Category        | Example Queries                          |
| --------------- | ---------------------------------------- |
| Web Development | react, nextjs, typescript, css, tailwind |
| Testing         | testing, jest, playwright, e2e           |
| DevOps          | deploy, docker, kubernetes, ci-cd        |
| Documentation   | docs, readme, changelog, api-docs        |
| Code Quality    | review, lint, refactor, best-practices   |
| Design          | ui, ux, design-system, accessibility     |
| Productivity    | workflow, automation, git                |

## Tips for Effective Searches

1. **Use specific keywords**: "react testing" is better than just "testing"
2. **Try alternative terms**: If "deploy" doesn't work, try "deployment" or "ci-cd"
3. **Check popular sources**: Many skills come from `vercel-labs/agent-skills` or `ComposioHQ/awesome-claude-skills`

## When No Skills Are Found

If no relevant skills exist:

1. Acknowledge that no existing skill was found
2. Offer to help with the task directly using your general capabilities
3. Suggest the user could create their own skill: a folder `skills/<skill-name>/` containing a `SKILL.md`

Example:

```
I searched for skills related to "xyz" but didn't find any matches.
I can still help you with this task directly! Would you like me to proceed?

If this is something you do often, you could create your own skill
in skills/my-xyz-skill/SKILL.md.
```
