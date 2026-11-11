// 无浏览器依赖的前端执行检查；模拟所需DOM节点，验证卡片、登记表及区间切换后的菱形。
const fs = require('fs'), path = require('path'), vm = require('vm'), assert = require('assert');
const root = path.resolve(__dirname, '../..');
const source = fs.readFileSync(path.join(root, 'alt_tpl.html'), 'utf8');
const data = JSON.parse(fs.readFileSync(path.join(root, 'data_alt.json'), 'utf8'));
const scripts = [...source.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(x => x[1].replace('__DATA__', '{}'));
scripts.forEach(script => new vm.Script(script));
class Element {
  constructor(tag, attrs = {}) { this.tag = tag; this.attrs = attrs; this.children = []; this.style = {}; this.innerHTML = ''; }
  appendChild(child) { this.children.push(child); return child; }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener() {}
}
const nodes = {};
const context = {D:data, NDX:data.series.ndx, S:data.series.temperature, AL:data.alerts,
  W:1000, RSTART:0, REND:data.series.dates.length-1,
  $:id => nodes[id] || (nodes[id] = new Element('div')),
  el:(tag, attrs) => new Element(tag, attrs), axisDates:dates => [0, dates.length-1], fmtDate:x => x};
// 实际页面S是温度序列，绘图在此仅需其日期长度；其他hover内容不会在此模拟。
if (!context.S) context.S = data.series.temperature_sell;
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('function drawNdx(){'), source.indexOf('/* TDC 条')), context);
vm.runInContext(source.slice(source.indexOf('const VV = D.validity'), source.indexOf('/* 实际利率条')), context);
assert(nodes.vtiles.innerHTML.includes('二次顶观察期'));
assert(nodes.vtiles.innerHTML.includes('未在观察期'));
for (const day of ['2026-06-02','2026-09-22','2026-10-16','2026-11-05']) assert(nodes.vtiles.innerHTML.includes(day));
assert(nodes['retest-sub'].innerHTML.includes('2026-11-12'));
assert(nodes['retest-sub'].innerHTML.includes('顶后 252'));
assert(nodes['retest-reg'].innerHTML.includes('尚无新进入的观察期'));
function markers(from, to) {
  context.RSTART = data.series.dates.findIndex(x => x >= from);
  context.REND = data.series.dates.findIndex(x => x > to) - 1;
  if (context.REND < 0) context.REND = data.series.dates.length - 1;
  nodes.cNdx = new Element('svg');
  vm.runInContext('drawNdx()', context);
  return nodes.cNdx.children.filter(x => x.attrs['data-retest-date']).map(x => x.attrs['data-retest-date']);
}
assert.deepStrictEqual(markers('2006-01-01','2026-11-09'), ['2012-10-09','2015-12-11','2025-01-10','2026-10-16']);
assert.deepStrictEqual(markers('2006-01-01','2016-12-31'), ['2012-10-09','2015-12-11']);
assert.deepStrictEqual(markers('2017-10-18','2026-11-09'), ['2025-01-10','2026-10-16']);
// 有新登记且当前观察中的分支。
const active = JSON.parse(JSON.stringify(data));
active.validity.retest.current = [{top:'2027-01-01',entry:'2027-04-01',trigger:null,end:null}];
active.validity.retest.registry.rows = [{event:'触发',top:'2027-01-01',entry:'2027-04-01',date:'2027-05-01',logged:'2027-05-01',fwd:-.05,low:-.12}];
context.D = active;
const branch = source.slice(source.indexOf('const VV = D.validity'), source.indexOf('/* 实际利率条')).replace('const VV =', 'const VV2 =').replace(/\bVV\b/g, 'VV2');
vm.runInContext(branch, context);
assert(nodes.vtiles.innerHTML.includes('观察中') && nodes.vtiles.innerHTML.includes('尚未触发'));
assert(nodes['retest-reg'].innerHTML.includes('-5.00%') && nodes['retest-reg'].innerHTML.includes('-12.00%'));
console.log(JSON.stringify({syntax:true,card:true,empty_registry:true,active_card:true,registry_table:true,markers_all:4,markers_out:2,markers_in:2,real_browser:false}));
