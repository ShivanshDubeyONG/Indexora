/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#F8FAFC',
        surface: {
          subtle: '#F2F8FF',
          secondary: '#E5F1FC',
          card: '#FFFFFF',
          elevated: '#FFFFFF',
        },
        accent: {
          DEFAULT: '#3986D9',
          hover: '#2563EB',
          subtle: '#E0EEFD',
          emerald: '#059669',
          amber: '#D97706',
          purple: '#7C3AED',
        },
        graphite: {
          DEFAULT: '#303943',
          dark: '#1E293B',
          muted: '#687583',
          subtle: '#94A3B8',
        },
        border: {
          DEFAULT: '#D5E2EF',
          focus: '#3986D9',
          subtle: '#E2E8F0',
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        heading: ['Montserrat', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        subtle: '0 1px 3px rgba(0, 0, 0, 0.03)',
        card: '0 2px 8px rgba(0, 0, 0, 0.04)',
        hover: '0 4px 16px rgba(57, 134, 217, 0.08)',
      },
    },
  },
  plugins: [],
};
