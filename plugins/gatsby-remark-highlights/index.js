const colorNames = require('color-name');

const presets = {
  yellow: ['#ffec9e', '#67531d'],
  green: ['#d8edcc', '#31543b'],
  blue: ['#d5e8fa', '#2e4b65'],
};

function getColor(value) {
  const name = value.toLowerCase();
  if (Object.prototype.hasOwnProperty.call(colorNames, name)) {
    return { name, rgb: colorNames[name] };
  }
  if (!/^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i.test(value)) return null;
  const hex =
    name.length === 4
      ? name
          .slice(1)
          .split('')
          .map((c) => c + c)
          .join('')
      : name.slice(1);
  return { name, rgb: [0, 2, 4].map((index) => parseInt(hex.slice(index, index + 2), 16)) };
}

function mix(rgb, base, amount) {
  return (
    '#' +
    rgb
      .map((channel, index) =>
        Math.round(channel * amount + base[index] * (1 - amount))
          .toString(16)
          .padStart(2, '0'),
      )
      .join('')
  );
}

function decorate(node) {
  let color = getColor('yellow');
  const first = node.children[0];
  if (first && first.type === 'text') {
    const separator = first.value.indexOf(':');
    const candidate = separator > 0 && getColor(first.value.slice(0, separator));
    if (candidate) {
      if (node.children.length === 1 && !first.value.slice(separator + 1).trim()) {
        node.type = 'text';
        node.value = `==${first.value}==`;
        delete node.children;
        return;
      }
      color = candidate;
      first.value = first.value.slice(separator + 1);
      if (!first.value) node.children.shift();
    }
  }
  const [light, dark] = presets[color.name] || [
    mix(color.rgb, [255, 255, 255], 0.28),
    mix(color.rgb, [35, 35, 38], 0.24),
  ];
  node.data = {
    hName: 'mark',
    hProperties: {
      className: ['text-highlight'],
      'data-highlight-color': color.name,
      style: `--highlight-light: ${light}; --highlight-dark: ${dark};`,
    },
  };
}

const underlineStyles = new Set(['solid', 'wavy', 'dashed', 'dotted', 'double']);
const underlineWeights = { thin: '1px', regular: '1.5px', thick: '3px' };

function luminance(rgb) {
  return rgb.reduce((total, channel, index) => {
    const c = channel / 255;
    const linear = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    return total + linear * [0.2126, 0.7152, 0.0722][index];
  }, 0);
}

// Keep bright lines visible on white and dark lines visible in the dark theme.
function strokeColor(rgb, background, base) {
  const bg = luminance(background);
  for (let step = 0; step <= 20; step++) {
    const hex = mix(rgb, base, 1 - step / 20);
    const channels = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16));
    const fg = luminance(channels);
    if ((Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05) >= 3) return hex;
  }
}

function decorateUnderline(node) {
  let style = 'solid';
  let weight = 'regular';
  let color = null;
  const first = node.children[0];
  const separator = first?.type === 'text' ? first.value.indexOf(':') : -1;
  if (separator > 0) {
    const options = first.value
      .slice(0, separator)
      .toLowerCase()
      .split(',')
      .map((s) => s.trim());
    const parsed = {};
    const valid = options.every((option) => {
      let key, value;
      if (underlineStyles.has(option)) [key, value] = ['style', option];
      else if (Object.prototype.hasOwnProperty.call(underlineWeights, option))
        [key, value] = ['weight', option];
      else {
        key = 'color';
        value = getColor(option);
        if (!value) return false;
      }
      if (parsed[key]) return false;
      parsed[key] = value;
      return true;
    });
    if (valid) {
      if (node.children.length === 1 && !first.value.slice(separator + 1).trim()) {
        node.type = 'text';
        node.value = `++${first.value}++`;
        delete node.children;
        return;
      }
      style = parsed.style || style;
      weight = parsed.weight || weight;
      color = parsed.color || color;
      first.value = first.value.slice(separator + 1);
      if (!first.value) node.children.shift();
    }
  }
  const colors = color
    ? ` --underline-light: ${strokeColor(
        color.rgb,
        [255, 255, 255],
        [0, 0, 0],
      )}; --underline-dark: ${strokeColor(color.rgb, [35, 35, 38], [255, 255, 255])};`
    : '';
  node.data = {
    hName: 'span',
    hProperties: {
      className: ['text-underline'],
      'data-underline-style': style,
      'data-underline-color': color?.name || 'currentColor',
      style: `--underline-style: ${style}; --underline-thickness: ${underlineWeights[weight]};${colors}`,
    },
  };
}

// A parser extension keeps Markdown inside highlights and underlines working, while escapes,
// inline code, fenced code, HTML attributes and link destinations stay literal.
function tokenize(effects, ok, nok) {
  let previous = this.previous;
  let marker;
  let name;
  let codeFence = 0;
  let ticks = 0;
  const closing = { partial: true, tokenize: checkClosing };
  return start;

  function start(code) {
    if ((code !== 61 && code !== 43) || previous === code) return nok(code);
    marker = code;
    name = code === 61 ? 'highlight' : 'underline';
    effects.enter(name);
    effects.enter(`${name}Sequence`);
    effects.consume(code);
    return second;
  }

  function second(code) {
    if (code !== marker) return nok(code);
    effects.consume(code);
    effects.exit(`${name}Sequence`);
    effects.enter(`${name}Text`, { contentType: 'text' });
    return first;
  }

  function first(code) {
    if (code === null || code === marker || code === 32 || code < 0) return nok(code);
    return body(code);
  }

  function body(code) {
    if (code === null || code < -2) return nok(code);
    if (code === marker && !codeFence && previous !== marker && previous !== 32 && previous >= 0) {
      return effects.check(closing, close, consume)(code);
    }
    if (code === 96) {
      ticks = 0;
      return backticks(code);
    }
    if (code === 92 && !codeFence) {
      effects.consume(code);
      previous = code;
      return escaped;
    }
    return consume(code);
  }

  function consume(code) {
    effects.consume(code);
    previous = code;
    return body;
  }

  function escaped(code) {
    if (code === null || code < -2) return nok(code);
    return consume(code);
  }

  function backticks(code) {
    if (code === 96) {
      effects.consume(code);
      previous = code;
      ticks++;
      return backticks;
    }
    if (!codeFence) codeFence = ticks;
    else if (codeFence === ticks) codeFence = 0;
    return body(code);
  }

  function checkClosing(check, yes, no) {
    return one;
    function one(code) {
      check.consume(code);
      return two;
    }
    function two(code) {
      if (code !== marker) return no(code);
      check.consume(code);
      return end;
    }
    function end(code) {
      return code === marker ? no(code) : yes(code);
    }
  }

  function close(code) {
    effects.exit(`${name}Text`);
    effects.enter(`${name}Sequence`);
    effects.consume(code);
    return closeSecond;
  }

  function closeSecond(code) {
    effects.consume(code);
    effects.exit(`${name}Sequence`);
    effects.exit(name);
    return ok;
  }
}

function remarkHighlights() {
  const data = this.data();
  (data.micromarkExtensions || (data.micromarkExtensions = [])).push({
    text: { 61: { tokenize }, 43: { tokenize } },
  });
  (data.fromMarkdownExtensions || (data.fromMarkdownExtensions = [])).push({
    enter: {
      highlight(token) {
        this.enter({ type: 'highlight', children: [] }, token);
      },
      underline(token) {
        this.enter({ type: 'underline', children: [] }, token);
      },
    },
    exit: {
      highlight(token) {
        const node = this.stack[this.stack.length - 1];
        this.exit(token);
        decorate(node);
      },
      underline(token) {
        const node = this.stack[this.stack.length - 1];
        this.exit(token);
        decorateUnderline(node);
      },
    },
  });
}

module.exports = ({ markdownAST }) => markdownAST;
module.exports.setParserPlugins = () => [remarkHighlights];
