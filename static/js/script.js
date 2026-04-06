// -------- 1. Utility Functions & Basic Interactions --------

function smoothScrollTo(targetY, duration = 800) {
  const startY = window.scrollY || window.pageYOffset;
  const startTime = performance.now();
  function frame(now) {
    const elapsed = (now - startTime) / duration;
    const t = Math.min(1, Math.max(0, elapsed));
    const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    window.scrollTo(0, startY + (targetY - startY) * eased);
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// Sidebar logic
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const body = document.body;

sidebarToggle.addEventListener('click', () => {
  const isCollapsed = body.classList.toggle('sidebar-collapsed');
  sidebarToggle.setAttribute('aria-expanded', String(!isCollapsed));
});

// Smooth scroll for anchor links
document.querySelectorAll('a[data-ease]').forEach((a) => {
  a.addEventListener('click', (e) => {
    e.preventDefault();
    const id = a.getAttribute('href');
    const el = document.querySelector(id);
    if (!el) return;
    const y = el.getBoundingClientRect().top + window.pageYOffset - 80;
    smoothScrollTo(y);
    if (window.innerWidth <= 860) {
      body.classList.add('sidebar-collapsed');
      sidebarToggle.setAttribute('aria-expanded', 'false');
    }
  });
});

// Back to top
const backToTop = document.getElementById('backToTop');
backToTop.addEventListener('click', () => smoothScrollTo(0));
window.addEventListener('scroll', () => {
  backToTop.classList.toggle('show', window.scrollY > 500);
});

// Reveal animation observer
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) entry.target.classList.add('show');
  });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal').forEach((el) => revealObserver.observe(el));

// TOC highlight spy
const tocLinks = document.querySelectorAll('.toc a');
const sectionIds = Array.from(tocLinks).map(a => a.getAttribute('href'));
const sectionEls = sectionIds.map(id => document.querySelector(id));

const spy = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = '#' + entry.target.id;
      tocLinks.forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === id);
      });
    }
  });
}, { rootMargin: '-20% 0% -60% 0%' });
sectionEls.forEach(el => el && spy.observe(el));


// -------- 2. Analysis Section: Fade In/Out Switch --------
document.addEventListener('DOMContentLoaded', () => {
  const methodButtons = document.querySelectorAll('.method-node');
  const contentWrapper = document.getElementById('methodContent');

  const analysisData = {
    analysis1: {
      badge: '01 \u00b7 Attention Dynamics',
      title: 'Layer-wise Attention Dynamics in Vision Encoders',
      image: 'static/image/attention_dynamics.webp',
      imageAlt: 'Analysis 1 visualization',
      html: `
        <p>
          In this study, we explore the internal attention dynamics of vision encoders.
          Through layer-wise analysis across multiple backbones,
          we reveal a consistent three-phase structure in how visual information is processed,
          independent of the model's architecture and scale.
        </p>
        <p>
          To quantify attention concentration, we introduced the \\(R^{(l)}\\) metric,
          defined as the ratio of the maximum attention score to attention entropy.
        </p>
        <p>Using this metric, we identified three distinct phases:</p>
        <ul>
          <li><strong>Phase 1 - Diffusion:</strong> In the early layers, attention remains <strong>broadly distributed</strong> across many visual tokens.</li>
          <li><strong>Phase 2 - Focus:</strong> Entering the intermediate layers, attention becomes <strong>highly concentrated</strong> on a small subset of specific tokens.</li>
          <li><strong>Phase 3 - Rediffusion:</strong> In the later layers, the previously concentrated attention distribution <strong>spreads out again</strong>.</li>
        </ul>
      `
    },
    analysis2: {
      badge: '02 \u00b7 Token Modulation',
      title: 'Effect of Phase-Specific Token Modulation',
      image: 'static/image/phase_masking.webp',
      imageAlt: 'Analysis 2 visualization',
      html: `
        <p>
          To identify the critical intervention point affecting hallucination behavior among the three observed phases,
          we adjusted the influence of "low-attention tokens" during the Diffusion, Focus, and Rediffusion phases
          by either suppressing them (Masking) or amplifying them (Inverse masking).
        </p>
        <p>The experimental results revealed the following key observations:</p>
        <ul>
          <li><strong>Phase 1 & 3 (Diffusion & Rediffusion):</strong> Modifying the influence of low-attention tokens produced only minor variations in the CHAIR metrics. Neither suppressing nor amplifying these tokens resulted in consistent improvements or degradations.</li>
          <li><strong>Phase 2 (Focus):</strong> Hallucination behavior proved highly sensitive to interventions in this phase. Suppressing low-attention tokens led to a consistent reduction in hallucinations, whereas amplifying their influence (inverse masking) significantly increased hallucination metrics relative to the baseline.</li>
          <li><strong>Invariant to Masking Strategies:</strong> Importantly, this reduction in hallucination during the Focus Phase was consistently observed across different token selection methods, including both simple rank-based Top-k and diversity-aware DPP strategies. This demonstrates that the phenomenon is robust and invariant to the specific masking methodology used.</li>
        </ul>
      `
    },
    analysis3: {
      badge: '03 \u00b7 VAR Dynamics',
      title: 'VAR Dynamics under Phase-Specific Masking',
      image: 'static/image/VAR.webp',
      imageAlt: 'Analysis 3 visualization',
      html: `
        <p>
          To investigate how phase-specific token modulation within the vision encoder affects the downstream language model,
          we analyzed the <a href="https://arxiv.org/abs/2411.16724" target="_blank" style="color: #2563eb; text-decoration: underline;">Visual Attention Ratio (VAR)</a>.
          VAR measures the extent to which generated tokens attend to visual inputs during decoding.
          A higher VAR indicates a stronger reliance on visual information,
          whereas a lower VAR suggests that the language model relies more heavily on language priors.
        </p>
        <p>The experimental results revealed the following key observations:</p>
        <ul>
          <li><strong>Significant VAR Increase in Phase 2:</strong> Masking applied during the focus phase produces a statistically significant increase in the mean VAR compared to the baseline.</li>
          <li><strong>Marginal Effects in Phase 1 & 3:</strong> In contrast, interventions applied during the diffusion or rediffusion phases result in only marginal changes.</li>
          <li><strong>Layer-wise Attention Shifts:</strong> VAR heatmaps (layer X head) further reveal increased visual attention in intermediate layers of the language model when focus phase masking is applied.</li>
        </ul>
      `
    },
    analysis4: {
      badge: '04 \u00b7 Qualitative Analysis',
      title: 'Qualitative Analysis using Ground-Truth Captions',
      image: 'static/image/GT_analysis.webp',
      imageAlt: 'Analysis 4 visualization',
      html: `
        <p>
          We analyzed the generated descriptions across different masking phases
          and compared them against Ground-Truth (GT) captions to observe how hallucinations manifest at the sentence level.
        </p>
        <ul>
          <li><strong>Inconsistent Outcomes (Phase 1 & 3):</strong> Masking during the diffusion or rediffusion phases fails to effectively control hallucinations, often preserving existing errors or introducing new ungrounded statements.</li>
          <li><strong>Reliable Mitigation (Phase 2):</strong> In contrast, masking applied specifically to the focus phase consistently replaces hallucinated statements (highlighted in red) with accurate descriptions that tightly align with the actual visual content.</li>
        </ul>
      `
    }
  };

  const updateAnalysisContent = (key) => {
    const data = analysisData[key];
    if (!data) return;
    document.getElementById('methodBadge').textContent = data.badge;
    document.getElementById('methodTitle').textContent = data.title;
    const fig = document.getElementById('methodFigure');
    fig.innerHTML = `<img src="${data.image}" alt="${data.imageAlt}" loading="lazy" class="analysis-image" />`;
    const copyBlock = document.getElementById('methodCopyBlock');
    copyBlock.innerHTML = data.html;
    // Re-typeset MathJax only when content contains LaTeX delimiters
    if (window.MathJax && window.MathJax.typesetPromise && /\\\(/.test(data.html)) {
      window.MathJax.typesetPromise([copyBlock]);
    }
  };

  methodButtons.forEach(btn => {
    btn.addEventListener('mouseenter', () => {
      if (btn.classList.contains('active')) return;
      methodButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const key = btn.getAttribute('data-method');
      contentWrapper.classList.remove('fade-in');
      setTimeout(() => {
        updateAnalysisContent(key);
        contentWrapper.classList.add('fade-in');
      }, 300);
    });
  });
});


// -------- 3. Qualitative Results Viewers (CHAIR & POPE) --------
document.addEventListener('DOMContentLoaded', () => {

  const chairData = [
    { src: 'static/image/llava7b_qualitive_chair.webp', model: 'LLaVA-1.5-7B' },
    { src: 'static/image/llava13b_qualitve_chair.webp', model: 'LLaVA-1.5-13B' },
    { src: 'static/image/Qwen_qualitive_chair.webp', model: 'Qwen-2.5-VL' },
    { src: 'static/image/Shikra_qualitive_chair.webp', model: 'Shikra-7B' },
    { src: 'static/image/Intern_qualitive_chair.webp', model: 'InternVL-2.5' }
  ];

  const popeData = [
    { src: 'static/image/llava7b_qualitive_pope.webp', model: 'LLaVA-1.5-7B' },
    { src: 'static/image/llava_13b_qualitive_pope.webp', model: 'LLaVA-1.5-13B' },
    { src: 'static/image/Qwen_qualitive_pope.webp', model: 'Qwen-2.5-VL' },
    { src: 'static/image/Shikra_qualitive_pope.webp', model: 'Shikra-7B' },
    { src: 'static/image/Intern_qualitive_pope.webp', model: 'InternVL-2.5' }
  ];

  function initQualViewer(pickerId, imageId, data) {
    const picker = document.getElementById(pickerId);
    const img = document.getElementById(imageId);

    if (!picker || !img) return;

    // Build tab buttons
    data.forEach((item, i) => {
      const btn = document.createElement('button');
      btn.role = 'tab';
      btn.className = 'chip';
      btn.textContent = item.model;
      btn.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
      btn.dataset.index = i;
      picker.appendChild(btn);
    });

    // Set initial image
    img.src = data[0].src;
    img.alt = data[0].model + ' qualitative result';

    // Tab click handler
    picker.addEventListener('click', (e) => {
      const btn = e.target.closest('[role="tab"]');
      if (!btn) return;
      const idx = parseInt(btn.dataset.index, 10);

      // Update tab UI
      picker.querySelectorAll('[role="tab"]').forEach(b => b.setAttribute('aria-selected', 'false'));
      btn.setAttribute('aria-selected', 'true');

      // Fade out, swap image, fade in
      img.classList.add('fade-out');
      setTimeout(() => {
        img.src = data[idx].src;
        img.alt = data[idx].model + ' qualitative result';
        img.classList.remove('fade-out');
      }, 250);
    });
  }

  initQualViewer('qualPickerChair', 'qualImageChair', chairData);
  initQualViewer('qualPickerPope', 'qualImagePope', popeData);
});
