/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: 'var(--bg-canvas)',
        surface: {
          DEFAULT: 'var(--bg-canvas)',
          subtle: 'var(--bg-surface-subtle)',
          secondary: 'var(--bg-surface-secondary)',
          card: 'var(--bg-surface-card)',
          elevated: 'var(--bg-surface-card)',
        },
        accent: {
          DEFAULT: 'var(--accent-primary)',
          hover: 'var(--accent-hover)',
          subtle: 'var(--accent-subtle)',
          emerald: '#10B981',
          amber: '#F59E0B',
          purple: '#8B5CF6',
        },
        graphite: {
          DEFAULT: 'var(--text-graphite)',
          dark: 'var(--text-graphite-dark)',
          muted: 'var(--text-graphite-muted)',
          subtle: 'var(--text-graphite-subtle)',
        },
        border: {
          DEFAULT: 'var(--border-color)',
          focus: 'var(--accent-primary)',
          subtle: 'var(--border-subtle)',
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        heading: ['Montserrat', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        subtle: '0 1px 3px rgba(0, 0, 0, 0.04)',
        card: '0 2px 8px rgba(0, 0, 0, 0.05)',
        hover: '0 4px 16px rgba(57, 134, 217, 0.12)',
        glow: '0 0 15px rgba(57, 134, 217, 0.25)',
      },
      keyframes: {
        computeFlash: {
          '0%': { borderColor: 'var(--accent-primary)', transform: 'scale(1.008)', boxShadow: '0 0 12px rgba(57, 134, 217, 0.35)' },
          '100%': { borderColor: 'var(--border-color)', transform: 'scale(1)', boxShadow: 'none' },
        },
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'compute-flash': 'computeFlash 0.5s ease-out',
        'fade-in': 'fadeIn 0.25s ease-out forwards',
      },
    },
  },
  plugins: [],
};
