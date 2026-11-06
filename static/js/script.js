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
      badge: '01 \u00b7 Attention\u2013Value Dynamics',
      title: 'Attention\u2013Value Dynamics Across Vision-Encoder Layers',
      image: 'static/image/attention_value_dynamics_combined.webp',
      imageAlt: 'Three-phase attention structure and low-attention value influence across vision-encoder layers',
      html: `
        <p>
          <strong>(a) Three-phase structure.</strong> We summarize layer-wise attention concentration as
          \\(R^{(\\ell)} = \\mathbb{E}_h[M^{(\\ell,h)}] / \\mathbb{E}_h[H^{(\\ell,h)}]\\),
          the ratio of the maximum attention score to attention entropy.
          Tracking \\(R^{(\\ell)}\\) across layers reveals a consistent three-phase pattern across LVLM backbones:
        </p>
        <ul>
          <li><strong>Phase 1 &middot; Diffusion:</strong> in early layers, attention is <strong>broadly distributed</strong> across visual tokens.</li>
          <li><strong>Phase 2 &middot; Focus:</strong> in intermediate layers, attention becomes <strong>sharply concentrated</strong> on a small subset of tokens, most clearly separating strongly and weakly supported tokens.</li>
          <li><strong>Phase 3 &middot; Rediffusion:</strong> in later layers, the concentrated pattern <strong>spreads out again</strong>.</li>
        </ul>
        <br />
        <p>
          <strong>(b) Low-attention value influence.</strong> However, low attention does not necessarily imply
          negligible downstream influence. Let \\(\\mathcal{L}^{(\\ell)}\\) be the bottom-25% visual tokens by
          received attention mass at layer \\(\\ell\\). We track three quantities across layers:
        </p>
        <ul>
          <li>\\(\\alpha_{\\mathcal{L}}^{(\\ell)}\\): total attention mass assigned to low-attention tokens.</li>
          <li>\\(\\eta_{\\mathcal{L}}^{(\\ell)}\\): their normalized value contribution to the attention output.</li>
          <li>\\(G_{\\mathcal{L}}^{(\\ell)} = \\eta_{\\mathcal{L}}^{(\\ell)} / (\\alpha_{\\mathcal{L}}^{(\\ell)} + \\epsilon)\\): value influence per unit attention mass.</li>
        </ul>
        <br />
        <p>
          During the focus phase, \\(\\alpha_{\\mathcal{L}}\\) drops sharply as attention concentrates,
          while \\(\\eta_{\\mathcal{L}}\\) remains non-negligible. The result is a large \\(G_{\\mathcal{L}}\\):
          low-attention tokens do <em>not</em> dominate the attention output, but their value content exerts a
          <strong>disproportionate influence relative to their small attention mass</strong>. This
          attention\u2013value mismatch motivates the phase-wise value intervention analysis.
        </p>
      `
    },
    analysis2: {
      badge: '02 \u00b7 Phase-wise Value Interventions',
      title: 'Phase-wise Value Interventions and Hallucination Behavior',
      image: 'static/image/phase_intervention.webp',
      imageAlt: 'Phase-wise value interventions and hallucination metrics',
      html: `
        <p>
          To test whether the focus-phase low-attention value signals are tied to hallucination, we
          run four diagnostic value interventions in each phase and measure CHAIR and F1 on captioning:
        </p>
        <ul>
          <li><strong>(A)</strong> replace high-attention values with zero vectors,</li>
          <li><strong>(B)</strong> replace low-attention values with zero vectors,</li>
          <li><strong>(C)</strong> replace low-attention values with the image-level mean value vector,</li>
          <li><strong>(D)</strong> replace low-attention values with low-attention values from another image.</li>
        </ul>
        <br />
        <p>The experiments reveal three observations:</p>
        <ul>
          <li><strong>Phase-localized effect:</strong> interventions in the diffusion and rediffusion phases produce only limited changes in hallucination metrics. In contrast, modifying low-attention values during the focus phase consistently reduces hallucination.</li>
          <li><strong>High-attention values carry grounded evidence:</strong> high-attention zero replacement (A) substantially degrades F1, indicating that strongly attended values encode visually grounded signals.</li>
          <li><strong>Low-attention values contain hallucination-sensitive signals:</strong> the three low-attention interventions (B, C, D) all reduce \\(\\text{CHAIR}_S\\) and \\(\\text{CHAIR}_I\\) while largely preserving F1. Since they share one common effect, removing or disrupting the original token-specific value content, the reduction is unlikely to be caused by a particular replacement value.</li>
        </ul>
      `
    },
    analysis3: {
      badge: '03 \u00b7 Token-Level Teacher Forcing',
      title: 'Token-Level Verification with Teacher Forcing',
      image: 'static/image/teacher_forcing_table.webp',
      imageAlt: 'Table of focus-phase low-attention value interventions on LLaVA-1.5-7B with CHAIR, F1, and token-level log-probability changes',
      html: `
        <p>
          Caption-level metrics show that hallucination decreases, but they do not directly reveal whether
          hallucinated object tokens themselves become less likely. For captions generated by the original
          model, we identify hallucinated and ground-truth object tokens, then recompute their
          log-probabilities under the same token sequence while applying focus-phase value interventions.
          For an object token span \\(y_s, \\dots, y_e\\), we compute the average log-probability:
        </p>
        <p class="method-formula">\\[ \\text{LP} =\\frac{1}{e-s+1} \\sum_{t=s}^{e} \\log p(y_t \\mid \\mathbf{x}, y_{&lt;t}) \\]</p>
        <p>
          The table reports results on LLaVA-1.5-7B, relative to the original model.
          <strong>Hallu LogP \\(\\Delta\\)</strong> is the average log-probability change of hallucinated object tokens,
          <strong>Drop Rate</strong> is the percentage of samples where that log-probability decreases, and
          <strong>GT LogP \\(\\Delta\\)</strong> is the corresponding change for ground-truth object tokens.
        </p>
        <p>The experiments reveal three observations:</p>
        <ul>
          <li><strong>Hallucinated tokens become less likely:</strong> all three low-attention interventions reduce their log-probability (\\(-0.136\\) to \\(-0.190\\)), in 63.5&ndash;68.2% of samples.</li>
          <li><strong>Ground-truth tokens are much less affected:</strong> for mean replacement, the change is only \\(-0.020\\), suggesting that the intervention does not globally suppress object generation.</li>
          <li><strong>Mean replacement provides the best trade-off:</strong> it achieves the lowest \\(\\text{CHAIR}_S\\) and \\(\\text{CHAIR}_I\\) with F1 nearly unchanged (74.2 &rarr; 74.1), so we use it as the default intervention.</li>
        </ul>
      `
    },
    analysis4: {
      badge: '04 \u00b7 Decoder-Side VAR',
      title: 'Decoder-Side Visual Attention under Focus-Phase Intervention',
      image: 'static/image/VAR_intervention.webp',
      imageAlt: 'Visual Attention Ratio under phase-wise value interventions',
      html: `
        <p>
          We further check how focus-phase value intervention affects the language model's reliance on
          visual tokens during decoding, using the
          <a href="https://arxiv.org/abs/2411.16724" target="_blank" style="color: #2563eb; text-decoration: underline;">Visual Attention Ratio (VAR)</a>:
          the total attention mass assigned by a generated token to image tokens at each LM layer-head.
        </p>
        <p>The experiments reveal three observations:</p>
        <ul>
          <li><strong>Significant VAR increase under focus-phase intervention:</strong> the image-level mean VAR rises from 0.0889 to <strong>0.1472</strong> (\\(p<0.001\\)), while diffusion- and rediffusion-phase interventions yield negligible changes (0.0876 and 0.0886).</li>
          <li><strong>Broader visual attention in intermediate LM layers:</strong> layer-head VAR heatmaps show increased visual-token attention across multiple intermediate layers and heads after focus-phase intervention.</li>
          <li><strong>Consistent with hallucination reduction:</strong> together with the CHAIR and teacher-forcing results, this suggests that neutralizing focus-phase low-attention values reduces hallucinated object likelihood and is accompanied by stronger decoder-side visual attention, while largely preserving ground-truth object information.</li>
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


// -------- 3. BibTeX Copy Button --------
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('bibtexCopy');
  const code = document.querySelector('#bibtex code');
  if (!btn || !code) return;

  let resetTimer;
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
    } catch {
      // Clipboard unavailable: select the text so it can be copied manually
      window.getSelection().selectAllChildren(code);
      return;
    }
    btn.classList.add('copied');
    btn.setAttribute('aria-label', 'Copied');
    btn.title = 'Copied';
    clearTimeout(resetTimer);
    resetTimer = setTimeout(() => {
      btn.classList.remove('copied');
      btn.setAttribute('aria-label', 'Copy BibTeX');
      btn.title = 'Copy BibTeX';
    }, 1500);
  });
});
