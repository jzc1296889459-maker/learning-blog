import { slug } from 'github-slugger'
import type { Blockquote, Link, Paragraph, PhrasingContent, Root, RootContent } from 'mdast'
import type { Plugin } from 'unified'

const CALLOUT_MARKER = /^\[!([a-z][\w-]*)\]([+-])?[ \t]*(.*?)(?:\r?\n|$)/i
const WIKILINK = /\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]/g

const CALLOUT_TITLES: Record<string, string> = {
  abstract: 'Abstract',
  example: 'Example',
  info: 'Info',
  note: 'Note',
  tip: 'Tip',
  warning: 'Warning'
}

const RL_NOTE_ID = '20260820---rl-math-foundation-01'
const RL_HEADING_ALIASES = {
  en: {
    'Conditional probability and conditional expectation':
      'Conditional probability and conditional expectation',
    'Probability notation versus ML/RL notation': 'Probability notation versus ML/RL notation',
    'Probability spaces, random variables, and expectation':
      'Preliminaries: measure-theoretic probability',
    'Transition kernels and measures on paths': 'Transition kernels and measures on paths'
  },
  zh: {
    'Conditional probability and conditional expectation': '条件概率与条件期望',
    'Probability notation versus ML/RL notation': '概率论记号与 ML/RL 记号的对照',
    'Probability spaces, random variables, and expectation': '预备知识：测度论概率',
    'Transition kernels and measures on paths': '转移核与路径空间上的测度'
  }
} as const

type ParentNode = Root | RootContent | PhrasingContent

function hasChildren(node: ParentNode): node is ParentNode & { children: RootContent[] } {
  return 'children' in node && Array.isArray(node.children)
}

function calloutTitle(type: string) {
  return CALLOUT_TITLES[type] ?? `${type.charAt(0).toUpperCase()}${type.slice(1)}`
}

export function transformCallout(node: Blockquote): boolean {
  const firstChild = node.children[0]
  if (firstChild?.type !== 'paragraph') return false

  const firstText = firstChild.children[0]
  if (firstText?.type !== 'text') return false

  const match = firstText.value.match(CALLOUT_MARKER)
  if (!match) return false

  const type = match[1].toLowerCase()
  const collapse = match[2]
  const title = match[3].trim() || calloutTitle(type)
  const remainder = firstText.value.slice(match[0].length)
  const bodyChildren: PhrasingContent[] = [...firstChild.children.slice(1)]

  if (remainder) bodyChildren.unshift({ type: 'text', value: remainder })
  if (bodyChildren[0]?.type === 'break') bodyChildren.shift()

  const titleParagraph: Paragraph = {
    type: 'paragraph',
    children: [{ type: 'text', value: title }],
    data: {
      hName: collapse ? 'summary' : 'p',
      hProperties: { className: ['obsidian-callout-title'] }
    }
  }
  const replacement: Blockquote['children'] = [titleParagraph]

  if (bodyChildren.length > 0) {
    replacement.push({ type: 'paragraph', children: bodyChildren })
  }

  node.children.splice(0, 1, ...replacement)
  node.data = {
    ...node.data,
    hName: collapse ? 'details' : 'aside',
    hProperties: {
      className: ['obsidian-callout', `obsidian-callout-${type}`],
      dataCallout: type,
      ...(collapse === '+' ? { open: true } : {})
    }
  }

  return true
}

function isEnglishFile(path = '') {
  return /\.en\.(?:md|mdx)$/i.test(path)
}

export function resolveWikiLink(target: string, filePath = ''): string {
  const hashIndex = target.indexOf('#')
  const page = hashIndex >= 0 ? target.slice(0, hashIndex).trim() : target.trim()
  const heading = hashIndex >= 0 ? target.slice(hashIndex + 1).trim() : ''

  if (!page) return heading ? `#${slug(heading)}` : '#'

  const language = isEnglishFile(filePath) ? 'en' : 'zh'
  const basePath = language === 'en' ? '/en/notes' : '/notes'

  if (page === 'Maths for RL') {
    const headingAlias = heading
      ? (RL_HEADING_ALIASES[language][
          heading as keyof (typeof RL_HEADING_ALIASES)[typeof language]
        ] ?? heading)
      : ''
    return `${basePath}/${RL_NOTE_ID}${headingAlias ? `#${slug(headingAlias)}` : ''}`
  }

  return `${basePath}/${slug(page)}${heading ? `#${slug(heading)}` : ''}`
}

function wikiLinkLabel(target: string, alias?: string) {
  if (alias) return alias

  const hashIndex = target.indexOf('#')
  const page = hashIndex >= 0 ? target.slice(0, hashIndex).trim() : target.trim()
  const heading = hashIndex >= 0 ? target.slice(hashIndex + 1).trim() : ''

  if (!page) return heading
  return heading ? `${page} — ${heading}` : page
}

export function splitWikiLinks(value: string, filePath = ''): PhrasingContent[] {
  const children: PhrasingContent[] = []
  let cursor = 0

  for (const match of value.matchAll(WIKILINK)) {
    const index = match.index ?? 0
    if (index > cursor) children.push({ type: 'text', value: value.slice(cursor, index) })

    const target = match[1].trim()
    const link: Link = {
      type: 'link',
      url: resolveWikiLink(target, filePath),
      children: [{ type: 'text', value: wikiLinkLabel(target, match[2]?.trim()) }]
    }
    children.push(link)
    cursor = index + match[0].length
  }

  if (cursor < value.length) children.push({ type: 'text', value: value.slice(cursor) })
  return children.length > 0 ? children : [{ type: 'text', value }]
}

function transformWikiLinks(node: ParentNode, filePath: string, insideLink = false) {
  if (!hasChildren(node)) return

  for (let index = 0; index < node.children.length; index += 1) {
    const child = node.children[index] as RootContent | PhrasingContent

    if (child.type === 'text' && !insideLink && child.value.includes('[[')) {
      const replacements = splitWikiLinks(child.value, filePath)
      node.children.splice(index, 1, ...(replacements as RootContent[]))
      index += replacements.length - 1
      continue
    }

    const nextInsideLink = insideLink || child.type === 'link' || child.type === 'linkReference'
    transformWikiLinks(child, filePath, nextInsideLink)
  }
}

function transformCallouts(node: ParentNode) {
  if (!hasChildren(node)) return

  for (const child of node.children) {
    if (child.type === 'blockquote') transformCallout(child)
    transformCallouts(child)
  }
}

export const remarkObsidian: Plugin<[], Root> = function () {
  return function (tree, file) {
    transformCallouts(tree)
    transformWikiLinks(tree, file.path ?? '')
  }
}

export default remarkObsidian
