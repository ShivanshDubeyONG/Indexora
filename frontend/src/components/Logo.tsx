import React from 'react';

interface LogoProps {
  className?: string;
  size?: number;
}

export const Logo: React.FC<LogoProps> = ({ className = '', size = 34 }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`flex-shrink-0 ${className}`}
    >
      <defs>
        <linearGradient id="idxGrad1" x1="4" y1="4" x2="36" y2="36" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#3986D9" />
          <stop offset="100%" stopColor="#2563EB" />
        </linearGradient>
        <linearGradient id="idxGrad2" x1="10" y1="20" x2="30" y2="20" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#60A5FA" />
          <stop offset="100%" stopColor="#93C5FD" />
        </linearGradient>
      </defs>

      {/* Outer rounded prism container with subtle border */}
      <rect x="2" y="2" width="36" height="36" rx="9" fill="url(#idxGrad1)" />

      {/* Orthogonal Vector Axis Grid */}
      <path
        d="M20 7L20 33"
        stroke="rgba(255, 255, 255, 0.25)"
        strokeWidth="1.5"
        strokeDasharray="2 2"
      />
      <path
        d="M7 20L33 20"
        stroke="rgba(255, 255, 255, 0.25)"
        strokeWidth="1.5"
        strokeDasharray="2 2"
      />

      {/* Hyper-dimensional Indexing Diamond / Hull */}
      <polygon
        points="20,10 30,20 20,30 10,20"
        fill="rgba(255, 255, 255, 0.12)"
        stroke="rgba(255, 255, 255, 0.75)"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />

      {/* Internal Vector Projections (Search Rays) */}
      <path
        d="M20 20L28 14"
        stroke="#FFFFFF"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path
        d="M20 20L13 27"
        stroke="#93C5FD"
        strokeWidth="1.5"
        strokeLinecap="round"
      />

      {/* Query Point Vertex (Glowing Core Centroid) */}
      <circle cx="20" cy="20" r="3.2" fill="#FFFFFF" />
      <circle cx="20" cy="20" r="5" stroke="rgba(255, 255, 255, 0.4)" strokeWidth="1" />

      {/* Nearest Neighbor Target Vertex */}
      <circle cx="28" cy="14" r="2.2" fill="#60A5FA" stroke="#FFFFFF" strokeWidth="1" />
      <circle cx="13" cy="27" r="1.8" fill="#93C5FD" />
      <circle cx="27" cy="25" r="1.5" fill="rgba(255, 255, 255, 0.7)" />
    </svg>
  );
};

