/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{astro,html,js,jsx,ts,tsx,md,mdx}'],
  theme: {
    extend: {
      fontFamily: {
        serif: ['Georgia', 'Cambria', 'Times New Roman', 'Times', 'serif'],
        sans: ['Inter', 'system-ui', 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace']
      },
      colors: {
        ink: '#0b0f14',
        paper: '#faf8f5',
        accent: '#0ea5a3',
        accent2: '#7c3aed'
      },
      typography: ({ theme }) => ({
        DEFAULT: {
          css: {
            maxWidth: '70ch',
            color: theme('colors.ink'),
            a: { color: theme('colors.accent'), textDecoration: 'underline', textUnderlineOffset: '2px' },
            h1: { fontWeight: '700', letterSpacing: '-0.02em' },
            h2: { fontWeight: '700', letterSpacing: '-0.02em' },
            code: { background: theme('colors.paper'), padding: '0.2rem 0.35rem', borderRadius: '0.25rem' }
          }
        }
      })
    }
  },
  plugins: [require('@tailwindcss/typography')]
};
