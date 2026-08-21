import { describe, expect, test } from 'bun:test'
import type { Blockquote } from 'mdast'

import { resolveWikiLink, splitWikiLinks, transformCallout } from './remark-obsidian'

describe('Obsidian callouts', () => {
  test('turns a callout blockquote into an aside and separates its title', () => {
    const callout: Blockquote = {
      type: 'blockquote',
      children: [
        {
          type: 'paragraph',
          children: [
            {
              type: 'text',
              value: '[!note] Definition - Generative model\nA generative model produces samples.'
            }
          ]
        }
      ]
    }

    expect(transformCallout(callout)).toBe(true)
    expect(callout.data?.hName).toBe('aside')
    expect(callout.data?.hProperties).toMatchObject({
      className: ['obsidian-callout', 'obsidian-callout-note'],
      dataCallout: 'note'
    })
    expect(callout.children).toHaveLength(2)
    expect(callout.children[0]).toMatchObject({
      type: 'paragraph',
      children: [{ type: 'text', value: 'Definition - Generative model' }]
    })
    expect(callout.children[1]).toMatchObject({
      type: 'paragraph',
      children: [{ type: 'text', value: 'A generative model produces samples.' }]
    })
  })

  test('turns a collapsed callout into a details element', () => {
    const callout: Blockquote = {
      type: 'blockquote',
      children: [
        {
          type: 'paragraph',
          children: [{ type: 'text', value: '[!abstract]- Rules of conditional expectation' }]
        }
      ]
    }

    transformCallout(callout)

    expect(callout.data?.hName).toBe('details')
    expect(callout.data?.hProperties).not.toHaveProperty('open')
    expect(callout.children[0].data?.hName).toBe('summary')
  })
})

describe('Obsidian wikilinks', () => {
  test('resolves same-page headings with the same slug format as Astro', () => {
    expect(resolveWikiLink('#ODE and SDE preliminaries', 'note.en.md')).toBe(
      '#ode-and-sde-preliminaries'
    )
    expect(resolveWikiLink('#ODE 与 SDE 预备知识', 'note.md')).toBe('#ode-与-sde-预备知识')
  })

  test('maps the known Maths for RL page to its bilingual route', () => {
    expect(
      resolveWikiLink(
        'Maths for RL#Probability spaces, random variables, and expectation',
        'note.en.md'
      )
    ).toBe('/en/notes/20260820---rl-math-foundation-01#preliminaries-measure-theoretic-probability')
    expect(
      resolveWikiLink(
        'Maths for RL#Probability spaces, random variables, and expectation',
        'note.md'
      )
    ).toBe('/notes/20260820---rl-math-foundation-01#预备知识测度论概率')
  })

  test('replaces wikilink text with mdast link nodes', () => {
    const children = splitWikiLinks('See [[#Flow matching]] next.', 'note.en.md')

    expect(children).toEqual([
      { type: 'text', value: 'See ' },
      {
        type: 'link',
        url: '#flow-matching',
        children: [{ type: 'text', value: 'Flow matching' }]
      },
      { type: 'text', value: ' next.' }
    ])
  })
})
