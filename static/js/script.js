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
      badge: '01 \u00b7 Three-Phase Structure',
      title: 'Three-Phase Attention Structure in Vision Encoders',
      image: 'static/image/attention_value_dynamics.webp',
      imageAlt: 'Three-phase attention structure across vision-encoder layers',
      html: `
        <p>
          We analyze how attention distributions evolve across vision-encoder layers and find a
          consistent depth-wise pattern across multiple LVLM backbones, independent of architecture
          or scale.
        </p>
        <p>
          We summarize layer-wise attention concentration as
          \\(R^{(\\ell)} = \\mathbb{E}_h[M^{(\\ell,h)}] / \\mathbb{E}_h[H^{(\\ell,h)}]\\),
          the ratio of the maximum attention score to attention entropy.
          Tracking \\(R^{(\\ell)}\\) across layers reveals three distinct phases:
        </p>
        <ul>
          <li><strong>Phase 1 &middot; Diffusion:</strong> in early layers, attention is <strong>broadly distributed</strong> across visual tokens.</li>
          <li><strong>Phase 2 &middot; Focus:</strong> in intermediate layers, attention becomes <strong>sharply concentrated</strong> on a small subset of tokens, most clearly separating strongly and weakly supported tokens.</li>
          <li><strong>Phase 3 &middot; Rediffusion:</strong> in later layers, the concentrated pattern <strong>spreads out again</strong>.</li>
        </ul>
      `
    },
    analysis2: {
      badge: '02 \u00b7 Attention\u2013Value Mismatch',
      title: 'Low-Attention Value Influence in the Focus Phase',
      image: 'static/image/low_att_value_influence.webp',
      imageAlt: 'Low-attention value influence across vision-encoder layers',
      html: `
        <p>
          Attention scores alone do not fully determine a token's downstream influence: self-attention
          aggregates value vectors weighted by attention, so a token with small attention mass can still
          shape the output if its value content remains non-negligible after aggregation.
        </p>
        <p>
          Let \\(\\mathcal{L}^{(\\ell)}\\) be the bottom-25% visual tokens by received attention mass at
          layer \\(\\ell\\). We track three quantities across layers:
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
          attention\u2013value mismatch motivates the focus-phase value intervention.
        </p>
      `
    },
    analysis3: {
      badge: '03 \u00b7 Phase-wise Value Interventions',
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
          <li><strong>Phase-localized effect:</strong> only focus-phase interventions consistently shift hallucination metrics; diffusion- and rediffusion-phase interventions barely move CHAIR.</li>
          <li><strong>High-attention values carry grounded evidence:</strong> high-attention zero replacement (A) substantially degrades F1, indicating that strongly attended values encode visually grounded signals.</li>
          <li><strong>Low-attention value content drives hallucination:</strong> the three low-attention interventions (B, C, D) all reduce \\(\\text{CHAIR}_S\\) and \\(\\text{CHAIR}_I\\) while largely preserving F1. Since they share only the effect of <em>neutralizing</em> the original token-specific values, the reduction is not an artifact of any one replacement vector.</li>
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
        <ul>
          <li><strong>Significant VAR increase under focus-phase intervention:</strong> the image-level mean VAR rises from 0.0889 to <strong>0.1472</strong> (\\(p<0.001\\)), while diffusion- and rediffusion-phase interventions yield negligible changes (0.0876 and 0.0886).</li>
          <li><strong>Broader visual attention in intermediate LM layers:</strong> layer-head VAR heatmaps show increased visual-token attention across multiple intermediate layers and heads after focus-phase intervention.</li>
          <li><strong>Consistent with hallucination reduction:</strong> together with the CHAIR and token-level teacher-forcing analyses, this indicates that neutralizing focus-phase low-attention values lets the decoder rely more on visual evidence rather than language priors.</li>
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


// -------- 3. Qualitative Results Viewer (CHAIR + POPE combined) --------
document.addEventListener('DOMContentLoaded', () => {

  const benchmarks = [
    { key: 'chair', label: 'CHAIR' },
    { key: 'pope', label: 'POPE' }
  ];

  const models = [
    { key: 'llava7b',  label: 'LLaVA-1.5-7B'  },
    { key: 'llava13b', label: 'LLaVA-1.5-13B' },
    { key: 'qwen',     label: 'Qwen-2.5-VL'   },
    { key: 'intern',   label: 'InternVL-2.5'  },
    { key: 'shikra',   label: 'Shikra-7B'     }
  ];

  // Image lookup: imageMap[benchmark][model] = src
  const imageMap = {
    chair: {
      llava7b:  'static/image/llava7b_qualitive_chair.webp',
      llava13b: 'static/image/llava13b_qualitve_chair.webp',
      qwen:     'static/image/Qwen_qualitive_chair.webp',
      shikra:   'static/image/Shikra_qualitive_chair.webp',
      intern:   'static/image/Intern_qualitive_chair.webp'
    },
    pope: {
      llava7b:  'static/image/llava7b_qualitive_pope.webp',
      llava13b: 'static/image/llava_13b_qualitive_pope.webp',
      qwen:     'static/image/Qwen_qualitive_pope.webp',
      shikra:   'static/image/Shikra_qualitive_pope.webp',
      intern:   'static/image/Intern_qualitive_pope.webp'
    }
  };

  const benchPicker = document.getElementById('qualPickerBench');
  const modelPicker = document.getElementById('qualPickerModel');
  const img = document.getElementById('qualImage');
  if (!benchPicker || !modelPicker || !img) return;

  let currentBench = benchmarks[0].key;
  let currentModel = models[0].key;

  function buildPicker(picker, items, currentKey) {
    items.forEach(item => {
      const btn = document.createElement('button');
      btn.role = 'tab';
      btn.className = 'chip';
      btn.textContent = item.label;
      btn.dataset.key = item.key;
      btn.setAttribute('aria-selected', item.key === currentKey ? 'true' : 'false');
      picker.appendChild(btn);
    });
  }

  function setSelected(picker, key) {
    picker.querySelectorAll('[role="tab"]').forEach(b => {
      b.setAttribute('aria-selected', b.dataset.key === key ? 'true' : 'false');
    });
  }

  function getBenchLabel(key) {
    const b = benchmarks.find(x => x.key === key);
    return b ? b.label : key;
  }

  function getModelLabel(key) {
    const m = models.find(x => x.key === key);
    return m ? m.label : key;
  }

  function updateImage() {
    const src = imageMap[currentBench] && imageMap[currentBench][currentModel];
    if (!src) return;
    img.classList.add('fade-out');
    setTimeout(() => {
      img.src = src;
      img.alt = `${getModelLabel(currentModel)} qualitative result on ${getBenchLabel(currentBench)}`;
      img.classList.remove('fade-out');
    }, 250);
  }

  buildPicker(benchPicker, benchmarks, currentBench);
  buildPicker(modelPicker, models, currentModel);

  // Initial image (no fade)
  img.src = imageMap[currentBench][currentModel];
  img.alt = `${getModelLabel(currentModel)} qualitative result on ${getBenchLabel(currentBench)}`;

  benchPicker.addEventListener('click', (e) => {
    const btn = e.target.closest('[role="tab"]');
    if (!btn) return;
    const key = btn.dataset.key;
    if (key === currentBench) return;
    currentBench = key;
    setSelected(benchPicker, key);
    updateImage();
  });

  modelPicker.addEventListener('click', (e) => {
    const btn = e.target.closest('[role="tab"]');
    if (!btn) return;
    const key = btn.dataset.key;
    if (key === currentModel) return;
    currentModel = key;
    setSelected(modelPicker, key);
    updateImage();
  });
});
