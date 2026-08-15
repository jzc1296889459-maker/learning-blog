import type { Config, IntegrationUserConfig, ThemeUserConfig } from 'astro-pure/types'

export const theme: ThemeUserConfig = {
  // === Basic configuration ===
  /** Title for your website. Will be used in metadata and as browser tab title. */
  title: 'My learning Blog',
  /** Will be used in index page & copyright declaration */
  author: 'James',
  /** Description metadata for your website. Can be used in page metadata. */
  description: '记录从零开始学习AI和Agent',
  /** The default favicon for your site which should be a path to an image in the `public/` directory. */
  favicon: '/favicon/favicon.ico',
  /** Specify the default language for this site. */
  locale: {
    lang: 'zh-CN',
    attrs: 'zh_CN',
    // Date locale
    dateLocale: 'zh-CN',
    dateOptions: {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    }
  },


  // === Global configuration ===
  titleDelimiter: '•',
  prerender: true,
  npmCDN: 'https://cdn.jsdelivr.net/npm',

  head: [],
  customCss: [],

  /** Configure the header of your site. */
  header: {
    menu: [
      { title: 'Blog', link: '/blog' },
      { title: 'Notes', link: '/notes' },

      { title: 'About', link: '/about' },

    ]
  },

  /** Configure the footer of your site. */
  footer: {
    links: [],
    /** Enable displaying a “Astro & Pure theme powered” link in your site’s footer. */
    credits: true,
    /** Optional details about the social media accounts for this site. */
    social: {
      github: 'https://github.com/jzc1296889459-maker'
    }
  },

  content: {
    externalLinksContent: ' ↗',
    /** Blog page size for pagination (optional) */
    blogPageSize: 8,
    externalLinkArrow: true, // show external link arrow
    share: []
  }
}

export const integ: IntegrationUserConfig = {
  // Page search runs on /api/search.json (see SiteSearch.astro); pagefind build hook disabled
  pagefind: false,
  // Required by astro-pure, but the Quote component is not used
  quote: {
    server: '',
    target: `() => ''`
  },
  // UnoCSS typography
  // See: https://unocss.dev/presets/typography
  typography: {
    class: 'prose text-base text-muted-foreground'
  },
  // A lightbox library that can add zoom effect
  // See: https://astro-pure.js.org/docs/integrations/others#medium-zoom
  mediumZoom: {
    enable: true, // disable it will not load the whole library
    selector: '.prose .zoomable',
    options: {
      className: 'zoomable'
    }
  },
  waline: {
    enable: false,
    server: '',
    additionalConfigs: {}
  }
}

const config = { ...theme, integ } as Config
export default config
