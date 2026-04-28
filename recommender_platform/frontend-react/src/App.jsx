import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, ShoppingBag, User, Star, TrendingUp, Sparkles, ChevronRight, X } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000/api/v1';
const FALLBACK_IMG = 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=800&q=80';

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

const App = () => {
  const [currentUser, setCurrentUser] = useState(() => localStorage.getItem('eliterec.user') || 'USER_0');
  const [personalizedRecs, setPersonalizedRecs] = useState([]);
  const [trendingItems, setTrendingItems] = useState([]);
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
      const [pRecs, tItems] = await Promise.allSettled([
        axios.get(`${API_BASE}/recommend/user/${currentUser}?limit=5`),
        axios.get(`${API_BASE}/recommend/trending?limit=8`)
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

      if (pRecs.status === 'rejected' && tItems.status === 'rejected') {
        setError('Could not reach the API. Make sure the backend is running on :8000 and try again.');
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
              {[...Array(10)].map((_, i) => (
                <option key={i} value={`USER_${i}`} className="bg-slate-900">User #{i}</option>
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
                    <p className="text-indigo-200/70 text-xs leading-relaxed">
                      We recommended this because of its high similarity to items you've viewed in {selectedProduct.category}. 
                      The rating of {selectedProduct.stars} shows strong customer satisfaction in this category.
                    </p>
                  </div>

                  <button className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-4 rounded-xl transition-all shadow-lg shadow-indigo-600/20">
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
