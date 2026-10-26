'use strict';

const copyButton = document.querySelector('[data-copy-citation]');
let copyResetTimer;
copyButton.addEventListener('click', async () => {
  window.clearTimeout(copyResetTimer);
  const citation = document.querySelector('#bibtex');
  const status = document.querySelector('.copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'BibTeX copied to clipboard.';
    copyButton.textContent = 'Copied!';
    copyResetTimer = window.setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2200);
  } catch {
    copyButton.textContent = 'Copy BibTeX';
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'BibTeX selected. Press Ctrl+C (or ⌘C) to copy.';
  }
});

// Keep the visible chapter in sync with reading position.
const siteHeader = document.querySelector('.site-header');
const chapterLinks = [...document.querySelectorAll('.nav-links a[href^="#"]')];
const chapters = chapterLinks.map(link => ({
  link,
  section: document.querySelector(link.getAttribute('href')),
})).filter(chapter => chapter.section);
let navigationFrame;
function updateNavigation() {
  navigationFrame = undefined;
  const readingLine = siteHeader.getBoundingClientRect().bottom + 96;
  let current;
  for (const chapter of chapters) {
    if (chapter.section.getBoundingClientRect().top <= readingLine) current = chapter;
  }
  // The last chapter may be too short to reach the reading line.
  if (window.scrollY > 0 && window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
    current = chapters[chapters.length - 1];
  }
  for (const chapter of chapters) {
    if (chapter === current) chapter.link.setAttribute('aria-current', 'location');
    else chapter.link.removeAttribute('aria-current');
  }
}
function scheduleNavigationUpdate() {
  if (navigationFrame === undefined) navigationFrame = window.requestAnimationFrame(updateNavigation);
}
window.addEventListener('scroll', scheduleNavigationUpdate, { passive: true });
window.addEventListener('resize', scheduleNavigationUpdate);
window.addEventListener('pageshow', scheduleNavigationUpdate);
document.fonts.ready.then(scheduleNavigationUpdate);
updateNavigation();
