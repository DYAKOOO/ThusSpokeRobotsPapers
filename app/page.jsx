'use client';

import { useState } from 'react';
import HUSKYQuiz from '../Feb2026/husky/HUSKYQuiz';
import HandelBotRealQuiz from '../Mar2026/HandelBotReal/HandelBotRealQuiz';
// NEW_QUIZ_IMPORT

const QUIZZES = [
  {
    slug: 'husky',
    title: 'HUSKY: Humanoid Skateboarding',
    authors: 'Han et al.',
    year: 2026,
    month: 'Feb',
    tags: ['locomotion', 'RL', 'AMP'],
    Component: HUSKYQuiz,
  },
  {
    slug: 'handelbot',
    title: 'HandelBot: Real-World Piano Playing',
    authors: 'Xie et al.',
    year: 2026,
    month: 'Mar',
    tags: ['dexterous', 'manipulation', 'adaptation'],
    Component: HandelBotRealQuiz,
  },
  // NEW_QUIZ_ENTRY
];  // END_QUIZZES

export default function Home() {
  const [active, setActive] = useState(null);

  if (active) {
    const { Component } = active;
    return (
      <div>
        <button
          onClick={() => setActive(null)}
          style={{ position: 'fixed', top: 16, left: 16, zIndex: 999,
                   background: '#1e1e2e', border: '1px solid #3f3f5a',
                   color: '#94a3b8', borderRadius: 8, padding: '6px 14px',
                   cursor: 'pointer', fontFamily: 'sans-serif', fontSize: '0.85rem' }}>
          ← All Papers
        </button>
        <Component />
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f', padding: '3rem 1.5rem',
                  fontFamily: "'Segoe UI', sans-serif" }}>
      <h1 style={{ color: '#f1f5f9', fontSize: '2rem', fontWeight: 700,
                   textAlign: 'center', marginBottom: '0.5rem' }}>
        Thus Spoke Robots
      </h1>
      <p style={{ color: '#64748b', textAlign: 'center', marginBottom: '3rem' }}>
        Interactive quizzes for robotics papers
      </p>
      <div style={{ maxWidth: 720, margin: '0 auto', display: 'grid', gap: '1rem' }}>
        {QUIZZES.map(q => (
          <div key={q.slug}
            onClick={() => setActive(q)}
            style={{ background: '#111118', border: '1px solid #2a2a3a',
                     borderRadius: 12, padding: '1.25rem 1.5rem', cursor: 'pointer' }}
            onMouseEnter={e => e.currentTarget.style.borderColor = '#4f46e5'}
            onMouseLeave={e => e.currentTarget.style.borderColor = '#2a2a3a'}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '1rem', marginBottom: 4 }}>
                  {q.title}
                </div>
                <div style={{ color: '#64748b', fontSize: '0.85rem' }}>
                  {q.authors} · {q.month} {q.year}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                {q.tags.map(tag => (
                  <span key={tag} style={{ background: '#1e1e2e', color: '#818cf8',
                    border: '1px solid #3f3f5a', borderRadius: 6,
                    padding: '2px 8px', fontSize: '0.75rem' }}>{tag}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
