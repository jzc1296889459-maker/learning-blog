# My Learning Blog

A Chinese-first bilingual blog recording my progress as I learn AI and AI agents from zero.

This site is based on [Joye Huang's blog](https://github.com/joyehuang/blog) and
[Astro Theme Pure](https://astro-pure.js.org/).

## Current scope

- Chinese and English homepages
- Blog
- Notes
- About
- Search, language switching, and dark mode
- Comments currently disabled

## Local development

Install dependencies:

```shell
bun install
```

Start the local website:

```shell
bun dev
```

Validate the project:

```shell
bun run check
bun test
```

Create a production build:

```shell
bun run build:checked
```

## Content organization

Chinese blog posts use `post.mdx`. Their English translations use `post.en.mdx`
in the same dated folder under `src/content/blog/`.

Notes are stored under `src/content/notes/`.

## Before deployment

- Replace the temporary `https://example.com` address in `astro.config.ts`.
- Confirm the site title, author, descriptions, and GitHub URL.
- Confirm the favicon and all images are my own or properly licensed.
- Keep Waline disabled unless I configure my own comment server.
- Keep analytics disabled unless I deliberately configure a privacy-appropriate service.
- Run `bun run build:checked` and review a Vercel preview deployment before publishing.
- Check Chinese and English routes, language switching, dark mode, search, RSS,
  sitemap, canonical URLs, and the 404 page.
- Confirm that no original author articles, images, QR codes, presentations,
  contact details, or personal information remain.

## License and content

The inherited source code is available under the Apache License 2.0. The original
`LICENSE` file is preserved.

The original author's articles, images, presentations, QR codes, and personal
content are not covered by the reusable template license and must not be reused here.

Unless stated otherwise, original blog posts and personal content are © 2026 James. All rights reserved.
