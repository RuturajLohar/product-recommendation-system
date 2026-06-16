import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, ShoppingBag, User, Star, TrendingUp, Sparkles, ChevronRight, X, BarChart3, Database, MousePointerClick, Layers, Brain, Network } from 'lucide-react';

// Dev: use same-origin `/api/v1` so Vite proxies to the backend (see vite.config.js).
// Prod / custom: set VITE_API_BASE, e.g. https://api.example.com/api/v1
const API_BASE =
  (import.meta.env.VITE_API_BASE && String(import.meta.env.VITE_API_BASE).trim()) ||
  (import.meta.env.DEV ? '/api/v1' : 'http://127.0.0.1:8000/api/v1');
const FALLBACK_IMG = 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=800&q=80';

const DEMO_USERS = [
  { id: 'USER_0', label: 'Speaker Fan' },
  { id: 'USER_1', label: 'Tech Buyer' },
  { id: 'USER_2', label: 'Budget Shopper' },
  { id: 'USER_3', label: 'Home Setup' },
  { id: 'CODEx_SMOKE_USER', label: 'Live Demo User' },
];

const HOW_IT_WORKS = [
  { icon: MousePointerClick, title: 'Track', text: 'Clicks and views are saved as user interactions.' },
  { icon: Brain, title: 'Embed', text: 'Product titles are converted into SBERT vectors.' },
  { icon: Network, title: 'Retrieve', text: 'Qdrant finds semantically similar products.' },
  { icon: Layers, title: 'Rank', text: 'Ranking signals sort and diversify the final list.' },
];

const formatNumber = (value) => new Intl.NumberFormat('en-US').format(value || 0);

const ProductCard = ({ product, onClick }) => (
  <motion.div
    whileHover={{ y: -5, scale: 1.02 }}
    className="glass rounded-2xl overflow-hidden cursor-pointer group"
    onClick={() => onClick(product)}
  >
    <div className="relative aspect-square bg-slate-800">
      <img 
        src={product.img_url || FALLBACK_IMG}
        alt={product.title}
        loading="lazy"
        onError={(e) => {
          e.currentTarget.src = FALLBACK_IMG;
        }}
        className="w-full h-full object-cover transition-transform group-hover:scale-110"
      />
      {typeof product.score === 'number' && product.score > 0 && product.score <= 1.0 && (
        <div className="absolute top-3 left-3 bg-indigo-600/90 backdrop-blur-md text-white text-[10px] px-2 py-1 rounded-full font-bold uppercase tracking-wider">
          {Math.round(product.score * 100)}% Match
        </div>
      )}
    </div>
    <div className="p-4">
      <p className="text-slate-400 text-xs uppercase tracking-widest mb-1">{product.category}</p>
      <h3 className="text-white font-semibold line-clamp-1 group-hover:text-indigo-400 transition-colors">{product.title}</h3>
      <div className="flex items-center justify-between mt-3">
        <p className="text-indigo-400 font-bold">${product.price}</p>
        <div className="flex items-center text-amber-400 text-xs">
          <Star size={12} fill="currentColor" />
          <span className="ml-1 text-slate-300">{product.stars}</span>
        </div>
      </div>
    </div>
  </motion.div>
);

const StatCard = ({ icon: Icon, label, value, detail }) => (
  <div className="glass rounded-2xl p-5">
    <div className="flex items-center justify-between">
      <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
        <Icon size={18} className="text-indigo-300" />
      </div>
      <span className="text-[10px] uppercase tracking-widest text-slate-500">Live</span>
    </div>
    <div className="text-2xl font-bold text-white mt-5">{value}</div>
    <div className="text-sm font-semibold text-slate-300 mt-1">{label}</div>
    {detail && <div className="text-xs text-slate-500 mt-2 line-clamp-1">{detail}</div>}
  </div>
);

const App = () => {
  const [currentUser, setCurrentUser] = useState(() => localStorage.getItem('eliterec.user') || DEMO_USERS[0].id);
  const [personalizedRecs, setPersonalizedRecs] = useState([]);
  const [trendingItems, setTrendingItems] = useState([]);
  const [analyticsSummary, setAnalyticsSummary] = useState(null);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);

  const personalizedRef = useRef(null);

  useEffect(() => {
    localStorage.setItem('eliterec.user', currentUser);
    axios.post(`${API_BASE}/users`, { external_id: currentUser }).catch(() => {});
    fetchData();
  }, [currentUser]);

  useEffect(() => {
    if (!selectedProduct) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, [selectedProduct]);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setSelectedProduct(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const trackEvent = async ({ type, asin }) => {
    try {
      await axios.post(`${API_BASE}/events`, { user_id: currentUser, asin, type });
    } catch {
      // best-effort
    }
  };

  const onProductClick = async (product) => {
    setSelectedProduct(product);
    if (product?.asin) trackEvent({ type: 'click', asin: product.asin });
  };

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [pRecs, tItems, stats] = await Promise.allSettled([
        axios.get(`${API_BASE}/recommend/user/${currentUser}?limit=5`),
        axios.get(`${API_BASE}/recommend/trending?limit=8`),
        axios.get(`${API_BASE}/analytics/summary`)
      ]);

      if (pRecs.status === 'fulfilled') {
        setPersonalizedRecs(pRecs.value.data.recommendations || []);
      } else {
        setPersonalizedRecs([]);
      }

      if (tItems.status === 'fulfilled') {
        setTrendingItems(tItems.value.data.recommendations || []);
      } else {
        setTrendingItems([]);
      }

      if (stats.status === 'fulfilled') {
        setAnalyticsSummary(stats.value.data);
      }

      if (pRecs.status === 'rejected' && tItems.status === 'rejected') {
        setError(
          'Could not reach the API. Start the backend on port 8000 (e.g. from recommender_platform: docker compose up -d db qdrant redis api), then click Retry. In dev, requests go through the Vite proxy at /api → http://127.0.0.1:8000.'
        );
      }
    } catch (error) {
      setError('Unexpected error while loading recommendations.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      setSearchOpen(false);
      return;
    }
    setSearching(true);
    setSearchOpen(true);
    const t = setTimeout(async () => {
      try {
        const res = await axios.get(`${API_BASE}/items`, { params: { q: searchQuery.trim(), limit: 8, offset: 0 } });
        setSearchResults(res.data || []);
      } catch {
        setSearchResults([]);
      } finally {
        setSearching(false);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [searchQuery]);

  const metricCards = useMemo(() => ([
    {
      icon: Database,
      label: 'Products indexed',
      value: formatNumber(analyticsSummary?.total_products),
      detail: analyticsSummary?.top_category?.name ? `Top category: ${analyticsSummary.top_category.name}` : 'Catalog ready',
    },
    {
      icon: User,
      label: 'Demo users',
      value: formatNumber(analyticsSummary?.total_users),
      detail: 'Switch users to see personalization change',
    },
    {
      icon: MousePointerClick,
      label: 'Interactions tracked',
      value: formatNumber(analyticsSummary?.total_interactions),
      detail: analyticsSummary?.most_interacted_product?.title || 'Click products to create signals',
    },
    {
      icon: BarChart3,
      label: 'Recommendation engine',
      value: 'Hybrid',
      detail: analyticsSummary?.active_strategy || 'SBERT + Qdrant + ranking',
    },
  ]), [analyticsSummary]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      {/* Navbar */}
      <nav className="sticky top-0 z-50 glass border-b border-white/5 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
            <Sparkles size={20} className="text-white" />
          </div>
          <span className="text-xl font-bold tracking-tighter text-white">ELITE<span className="text-indigo-500">REC</span></span>
        </div>
        
        <div className="hidden md:block relative w-96">
          <div className="flex items-center bg-white/5 border border-white/10 rounded-full px-4 py-1.5 w-full focus-within:border-indigo-500/50 transition-colors">
          <Search size={18} className="text-slate-500" />
          <input 
            type="text" 
            placeholder="Search products..." 
            className="bg-transparent border-none outline-none ml-2 w-full text-sm"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={() => searchQuery.trim() && setSearchOpen(true)}
          />
          </div>

          {searchOpen && (
            <div className="absolute top-full mt-2 w-full glass rounded-2xl overflow-hidden border border-white/10">
              {searching ? (
                <div className="p-4 text-sm text-slate-400">Searching...</div>
              ) : searchResults.length ? (
                <div className="max-h-96 overflow-auto">
                  {searchResults.map((prod) => (
                    <button
                      key={prod.asin}
                      onClick={() => {
                        setSearchOpen(false);
                        setSearchQuery('');
                        onProductClick(prod);
                      }}
                      className="w-full text-left px-4 py-3 hover:bg-white/5 flex items-center gap-3"
                    >
                      <img
                        src={prod.img_url || FALLBACK_IMG}
                        alt={prod.title}
                        loading="lazy"
                        onError={(e) => {
                          e.currentTarget.src = FALLBACK_IMG;
                        }}
                        className="w-10 h-10 rounded-lg object-cover bg-slate-900"
                      />
                      <div className="min-w-0">
                        <div className="text-sm font-semibold text-white line-clamp-1">{prod.title}</div>
                        <div className="text-xs text-slate-400 line-clamp-1">{prod.category}</div>
                      </div>
                      <div className="ml-auto text-sm font-bold text-indigo-300">${prod.price}</div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="p-4 text-sm text-slate-400">No results.</div>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2 bg-indigo-500/10 px-3 py-1.5 rounded-full border border-indigo-500/20">
            <User size={16} className="text-indigo-400" />
            <select 
              className="bg-transparent text-xs font-semibold outline-none text-indigo-300"
              value={currentUser}
              onChange={(e) => setCurrentUser(e.target.value)}
            >
              {DEMO_USERS.map((user) => (
                <option key={user.id} value={user.id} className="bg-slate-900">{user.label}</option>
              ))}
            </select>
          </div>
          <ShoppingBag size={20} className="text-slate-400 hover:text-white cursor-pointer transition-colors" />
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-10 space-y-16">
        {error && (
          <div className="glass border border-rose-500/20 bg-rose-500/5 rounded-2xl p-4 flex items-center justify-between">
            <div className="text-sm text-rose-200">{error}</div>
            <button
              onClick={fetchData}
              className="text-sm font-semibold text-rose-200 hover:text-white"
            >
              Retry
            </button>
          </div>
        )}

        <section>
          <div className="flex items-end justify-between mb-6">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-indigo-300 font-bold">Project Dashboard</p>
              <h1 className="text-4xl font-bold text-white tracking-tight mt-2">Hybrid Recommendation Demo</h1>
              <p className="text-slate-400 mt-2 max-w-2xl">
                A compact college-project demo showing catalog search, event tracking, vector retrieval, ranking, and personalized recommendations in one flow.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {metricCards.map((card) => (
              <StatCard key={card.label} {...card} />
            ))}
          </div>
        </section>

        {/* Hero / Personalized */}
        <section ref={personalizedRef}>
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-bold text-white tracking-tight flex items-center">
                Picked for <span className="gradient-text ml-2">You</span>
                <Sparkles size={24} className="ml-3 text-indigo-500 animate-pulse-slow" />
              </h2>
              <p className="text-slate-400 mt-1">Based on your recent interest in {personalizedRecs[0]?.category || 'Tech'}</p>
            </div>
            <button
              onClick={() => personalizedRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
              className="flex items-center text-sm font-semibold text-indigo-400 hover:text-indigo-300 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              disabled={!personalizedRecs.length}
            >
              View all <ChevronRight size={16} className="ml-1" />
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
            {loading ? (
              [...Array(5)].map((_, i) => <div key={i} className="aspect-[3/4] bg-white/5 animate-pulse rounded-2xl" />)
            ) : (
              personalizedRecs.length ? (
                personalizedRecs.map((prod) => (
                  <ProductCard key={prod.asin} product={prod} onClick={onProductClick} />
                ))
              ) : (
                <div className="col-span-2 md:col-span-5 glass rounded-2xl p-6 text-slate-300">
                  <div className="text-white font-semibold">No personalized recommendations yet.</div>
                  <div className="text-sm text-slate-400 mt-1">Click a few products below to build your history.</div>
                </div>
              )
            )}
          </div>
        </section>

        <section>
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-2xl font-bold text-white tracking-tight">How It Works</h2>
              <p className="text-slate-400 mt-1">The full recommendation pipeline, simplified for presentation.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {HOW_IT_WORKS.map(({ icon: Icon, title, text }, index) => (
              <div key={title} className="glass rounded-2xl p-5">
                <div className="flex items-center justify-between mb-5">
                  <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                    <Icon size={18} className="text-indigo-300" />
                  </div>
                  <span className="text-xs text-slate-500 font-bold">0{index + 1}</span>
                </div>
                <div className="text-white font-bold">{title}</div>
                <p className="text-sm text-slate-400 mt-2">{text}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Trending */}
        <section>
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold text-white tracking-tight flex items-center">
              Trending <span className="text-indigo-500 ml-2">Now</span>
              <TrendingUp size={20} className="ml-3 text-emerald-500" />
            </h2>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {loading ? (
              [...Array(8)].map((_, i) => <div key={i} className="aspect-square bg-white/5 animate-pulse rounded-2xl" />)
            ) : (
              trendingItems.map((prod) => (
                <ProductCard key={prod.asin} product={prod} onClick={onProductClick} />
              ))
            )}
          </div>
        </section>
      </main>

      {/* Modal for Detail */}
      <AnimatePresence>
        {selectedProduct && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedProduct(null)}
              className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="relative glass max-w-4xl w-full rounded-3xl overflow-hidden shadow-2xl flex flex-col md:flex-row max-h-[90vh]"
            >
              <button
                onClick={() => setSelectedProduct(null)}
                className="absolute top-4 right-4 z-10 w-9 h-9 rounded-full bg-slate-900/70 hover:bg-slate-900 text-slate-200 flex items-center justify-center"
                aria-label="Close"
              >
                <X size={18} />
              </button>
              <div className="md:w-1/2 aspect-square md:aspect-auto bg-slate-900">
                <img 
                  src={selectedProduct.img_url || FALLBACK_IMG}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.currentTarget.src = FALLBACK_IMG;
                  }}
                />
              </div>
              <div className="md:w-1/2 p-8 flex flex-col overflow-y-auto">
                <div className="flex-1">
                  <p className="text-indigo-500 font-bold uppercase tracking-widest text-xs mb-2">{selectedProduct.category}</p>
                  <h2 className="text-3xl font-bold text-white mb-4">{selectedProduct.title}</h2>
                  <div className="flex items-center space-x-4 mb-6">
                    <p className="text-2xl font-bold text-white">${selectedProduct.price}</p>
                    {selectedProduct.is_best_seller && (
                      <div className="flex items-center bg-emerald-500/10 text-emerald-500 px-2 py-1 rounded text-xs font-bold">
                        <TrendingUp size={12} className="mr-1" /> Best Seller
                      </div>
                    )}
                  </div>
                  
                  <div className="bg-indigo-500/10 rounded-xl p-4 border border-indigo-500/20 mb-8">
                    <div className="flex items-center text-indigo-300 text-sm font-semibold mb-2">
                      <Sparkles size={16} className="mr-2" /> AI Recommendation Insight
                    </div>
                    <p className="text-indigo-200/70 text-xs leading-relaxed italic">
                      {selectedProduct.explanation_text || `We recommended this because of its high similarity to items you've viewed in ${selectedProduct.category}. The rating of ${selectedProduct.stars} shows strong customer satisfaction.`}
                    </p>
                  </div>

                  <button
                    onClick={async () => {
                      if (!selectedProduct?.asin) return;
                      await trackEvent({ type: 'purchase', asin: selectedProduct.asin });
                      fetchData();
                    }}
                    className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-4 rounded-xl transition-all shadow-lg shadow-indigo-600/20"
                  >
                    Add to Cart
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;
