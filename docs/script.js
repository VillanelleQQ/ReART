const carousels = document.querySelectorAll("[data-carousel]");

carousels.forEach((carousel) => {
  const slides = Array.from(carousel.querySelectorAll(".slide"));
  const previous = carousel.querySelector("[data-prev]");
  const next = carousel.querySelector("[data-next]");
  const counter = carousel.querySelector("[data-counter]");
  let index = 0;

  const render = () => {
    slides.forEach((slide, slideIndex) => {
      slide.classList.toggle("is-active", slideIndex === index);
    });
    counter.textContent = `${index + 1} / ${slides.length}`;
  };

  previous.addEventListener("click", () => {
    index = (index - 1 + slides.length) % slides.length;
    render();
  });

  next.addEventListener("click", () => {
    index = (index + 1) % slides.length;
    render();
  });

  carousel.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      previous.click();
    }
    if (event.key === "ArrowRight") {
      next.click();
    }
  });

  carousel.tabIndex = 0;
  render();
});

const copyButtons = document.querySelectorAll("[data-copy-target]");

const copyText = async (text) => {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
};

copyButtons.forEach((button) => {
  const defaultLabel = button.textContent;
  const targetId = button.dataset.copyTarget;
  const target = document.getElementById(targetId);

  if (!target) {
    return;
  }

  button.addEventListener("click", async () => {
    try {
      await copyText(target.textContent.trim());
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = defaultLabel;
      }, 1800);
    } catch {
      button.textContent = "Failed";
      window.setTimeout(() => {
        button.textContent = defaultLabel;
      }, 1800);
    }
  });
});
