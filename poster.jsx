import React from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const BatteryPoster = () => {
  // Data
  const priceData = [
    { year: '2010', price: 1200 },
    { year: '2013', price: 650 },
    { year: '2016', price: 290 },
    { year: '2019', price: 200 },
    { year: '2022', price: 150 },
    { year: '2025', price: 115 }
  ];

  const emissionsData = [
    { source: 'Wind+Batt', value: 82, color: '#8B5CF6' },
    { source: 'Solar+Batt', value: 116, color: '#3B82F6' },
    { source: 'Gas+CCS', value: 170, color: '#6B7280' },
    { source: 'Coal+CCS', value: 220, color: '#4B5563' },
    { source: 'Gas', value: 490, color: '#9CA3AF' },
    { source: 'Coal', value: 820, color: '#1F2937' }
  ];

  const jobData = [
    { year: '2020', jobs: 5 },
    { year: '2023', jobs: 8 },
    { year: '2025', jobs: 10 },
    { year: '2027', jobs: 16 },
    { year: '2030', jobs: 30 }
  ];

  return (
    <div className="w-[1600px] h-[1133px] bg-white" style={{ fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }}>
      
      {/* Header */}
      <div className="h-[90px] bg-gradient-to-r from-cyan-500 via-cyan-400 to-blue-500 px-8 flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-white tracking-tight">Energy Storage with Batteries: Overview</h1>
          <p className="text-sm text-white/90 mt-1">Aaron Cao • Xianwei Xu • Daniel Dutulescu • Yihan Yang • Vishnu Barath  |  Group B</p>
        </div>
        <img src="/mnt/user-data/uploads/kindpng_4496018.png" alt="UCL Logo" className="h-16 object-contain" />
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-3 p-5 h-[990px]">
        
        {/* Left Column - 4 cols */}
        <div className="col-span-4 space-y-3">
          
          {/* Introduction */}
          <div className="bg-gradient-to-br from-blue-50 to-white p-4 rounded-lg border border-blue-200 shadow-sm">
            <h2 className="text-2xl font-bold text-gray-900 mb-3 flex items-center">
              <span className="w-1 h-6 bg-blue-600 mr-3"></span>
              Introduction
            </h2>
            <ul className="space-y-2 text-[13px] text-gray-800 leading-snug">
              <li className="flex items-start">
                <span className="text-blue-600 mr-2 font-bold">•</span>
                <span>Fossil fuels supply 80% of global energy but drive climate instability<sup className="text-blue-600">13,14</sup></span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2 font-bold">•</span>
                <span><strong>Battery storage</strong> enables renewable transition by storing excess solar/wind<sup className="text-blue-600">1,2</sup></span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2 font-bold">•</span>
                <span>Eliminates fossil backup, stabilizes grids, reduces costs<sup className="text-blue-600">3,4</sup></span>
              </li>
            </ul>
          </div>

          {/* Battery Price Decline */}
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <h2 className="text-xl font-bold text-gray-900 mb-2 flex items-center">
              <span className="w-1 h-5 bg-purple-600 mr-3"></span>
              Cost Evolution<sup className="text-xs text-blue-600">2,3</sup>
            </h2>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={priceData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis dataKey="year" stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} />
                <YAxis stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} />
                <Line type="monotone" dataKey="price" stroke="#8B5CF6" strokeWidth={3} dot={{ fill: '#8B5CF6', r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <div className="bg-gradient-to-br from-purple-100 to-purple-50 p-3 rounded-lg">
                <div className="text-2xl font-black text-purple-900">90%</div>
                <div className="text-xs text-gray-700">Price Reduction</div>
              </div>
              <div className="bg-gradient-to-br from-blue-100 to-blue-50 p-3 rounded-lg">
                <div className="text-2xl font-black text-blue-900">$115</div>
                <div className="text-xs text-gray-700">Per kWh (2025)</div>
              </div>
            </div>
          </div>

          {/* ROI */}
          <div className="bg-gradient-to-br from-green-50 to-white p-4 rounded-lg border border-green-200 shadow-sm">
            <h2 className="text-xl font-bold text-gray-900 mb-3 flex items-center">
              <span className="w-1 h-5 bg-green-600 mr-3"></span>
              Return on Investment<sup className="text-xs text-blue-600">4</sup>
            </h2>
            <div className="space-y-2 text-[13px]">
              <div className="flex justify-between items-center bg-white p-2 rounded border border-gray-200">
                <span className="font-semibold text-gray-800">Residential</span>
                <span className="text-green-700 font-bold">5-10 years</span>
              </div>
              <div className="flex justify-between items-center bg-white p-2 rounded border border-gray-200">
                <span className="font-semibold text-gray-800">Commercial</span>
                <span className="text-green-700 font-bold">3-6 years</span>
              </div>
              <div className="flex justify-between items-center bg-white p-2 rounded border border-gray-200">
                <span className="font-semibold text-gray-800">Utility Scale</span>
                <span className="text-green-700 font-bold">3-4 years</span>
              </div>
              <div className="bg-green-100 p-2 rounded text-xs text-gray-800 mt-2">
                <strong>Post-payback:</strong> 15-25+ years profit generation. US Federal ITC (30%) reduces payback by 2-3 years.
              </div>
            </div>
          </div>

          {/* Job Growth */}
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <h2 className="text-xl font-bold text-gray-900 mb-2 flex items-center">
              <span className="w-1 h-5 bg-blue-600 mr-3"></span>
              Job Growth Projection<sup className="text-xs text-blue-600">5</sup>
            </h2>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={jobData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis dataKey="year" stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} />
                <YAxis stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} />
                <Line type="monotone" dataKey="jobs" stroke="#3B82F6" strokeWidth={3} dot={{ fill: '#3B82F6', r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <div className="bg-blue-50 p-2 rounded">
                <div className="font-bold text-gray-900">US: 25 facilities</div>
                <div className="text-gray-700">$13B invested, 10K+ jobs<sup className="text-blue-600">5,12</sup></div>
              </div>
              <div className="bg-blue-50 p-2 rounded">
                <div className="font-bold text-gray-900">Tesla Giga Berlin</div>
                <div className="text-gray-700">6K→20K jobs by 2027<sup className="text-blue-600">12</sup></div>
              </div>
            </div>
          </div>
        </div>

        {/* Middle Column - 4 cols */}
        <div className="col-span-4 space-y-3">
          
          {/* Emissions Comparison */}
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <h2 className="text-2xl font-bold text-gray-900 mb-3 flex items-center">
              <span className="w-1 h-6 bg-purple-600 mr-3"></span>
              Lifecycle CO₂ Emissions<sup className="text-xs text-blue-600">6,7</sup>
            </h2>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={emissionsData} layout="vertical" margin={{ top: 5, right: 30, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis type="number" stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} label={{ value: 'g CO₂/kWh', position: 'insideBottom', offset: -5, style: { fontSize: 11 } }} />
                <YAxis dataKey="source" type="category" stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} width={80} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {emissionsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-3 gap-2 mt-3">
              <div className="bg-purple-100 p-2 rounded text-center">
                <div className="text-xl font-black text-purple-900">86%</div>
                <div className="text-xs text-gray-700">vs Coal+CCS</div>
              </div>
              <div className="bg-blue-100 p-2 rounded text-center">
                <div className="text-xl font-black text-blue-900">90%</div>
                <div className="text-xs text-gray-700">vs Gas+CCS</div>
              </div>
              <div className="bg-green-100 p-2 rounded text-center">
                <div className="text-xl font-black text-green-900">&lt;1yr</div>
                <div className="text-xs text-gray-700">Mfg Payback<sup className="text-blue-600">6</sup></div>
              </div>
            </div>
          </div>

          {/* Next-Gen Battery Comparison */}
          <div className="bg-gradient-to-br from-purple-50 to-white p-4 rounded-lg border border-purple-200 shadow-sm">
            <h2 className="text-2xl font-bold text-gray-900 mb-3 flex items-center">
              <span className="w-1 h-6 bg-purple-600 mr-3"></span>
              Next-Gen Battery Technologies<sup className="text-xs text-blue-600">8,9,10</sup>
            </h2>
            
            {/* Comparison Grid */}
            <div className="grid grid-cols-4 gap-2 mb-3">
              {/* Li-ion */}
              <div className="bg-white rounded-lg p-3 border border-gray-300">
                <div className="text-center mb-2">
                  <div className="text-xs font-bold text-gray-600">CURRENT</div>
                  <div className="text-lg font-black text-gray-900">Li-ion</div>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Energy:</span>
                    <span className="font-bold text-blue-600">⚡⚡⚡⚡⚡</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cycles:</span>
                    <span className="font-bold">⟳⟳⟳⟳</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cost:</span>
                    <span className="font-bold">$$</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Safety:</span>
                    <span className="font-bold">⚠️⚠️⚠️</span>
                  </div>
                </div>
                <div className="mt-2 text-xs bg-gray-100 p-2 rounded">
                  <strong>Status:</strong> Commercial<br/>
                  <strong>Issue:</strong> Rare materials, thermal runaway risk
                </div>
              </div>

              {/* Na-ion */}
              <div className="bg-white rounded-lg p-3 border-2 border-blue-400">
                <div className="text-center mb-2">
                  <div className="text-xs font-bold text-blue-600">EMERGING</div>
                  <div className="text-lg font-black text-gray-900">Na-ion</div>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Energy:</span>
                    <span className="font-bold text-blue-600">⚡⚡⚡⚡</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cycles:</span>
                    <span className="font-bold">⟳⟳⟳⟳</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cost:</span>
                    <span className="font-bold text-green-600">$</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Safety:</span>
                    <span className="font-bold text-green-600">✓✓✓✓</span>
                  </div>
                </div>
                <div className="mt-2 text-xs bg-blue-50 p-2 rounded">
                  <strong>Status:</strong> Commercial (CATL)<sup className="text-blue-600">10</sup><br/>
                  <strong>Win:</strong> Abundant Na, safer chemistry
                </div>
              </div>

              {/* Solid-State */}
              <div className="bg-white rounded-lg p-3 border-2 border-purple-400">
                <div className="text-center mb-2">
                  <div className="text-xs font-bold text-purple-600">NEXT-GEN</div>
                  <div className="text-lg font-black text-gray-900">Solid-State</div>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Energy:</span>
                    <span className="font-bold text-purple-600">⚡⚡⚡⚡⚡⚡</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cycles:</span>
                    <span className="font-bold text-purple-600">⟳⟳⟳⟳⟳</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cost:</span>
                    <span className="font-bold">$$$</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Safety:</span>
                    <span className="font-bold text-green-600">✓✓✓✓✓</span>
                  </div>
                </div>
                <div className="mt-2 text-xs bg-purple-50 p-2 rounded">
                  <strong>Status:</strong> Development<br/>
                  <strong>Win:</strong> 2× energy, no fire risk
                </div>
              </div>

              {/* Zn-ion */}
              <div className="bg-white rounded-lg p-3 border-2 border-cyan-400">
                <div className="text-center mb-2">
                  <div className="text-xs font-bold text-cyan-600">EMERGING</div>
                  <div className="text-lg font-black text-gray-900">Zn-ion</div>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Energy:</span>
                    <span className="font-bold text-blue-600">⚡⚡⚡</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cycles:</span>
                    <span className="font-bold">⟳⟳⟳</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cost:</span>
                    <span className="font-bold text-green-600">$</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Safety:</span>
                    <span className="font-bold text-green-600">✓✓✓✓✓</span>
                  </div>
                </div>
                <div className="mt-2 text-xs bg-cyan-50 p-2 rounded">
                  <strong>Status:</strong> R&D breakthrough<sup className="text-blue-600">9</sup><br/>
                  <strong>Win:</strong> Abundant Zn, non-toxic
                </div>
              </div>
            </div>

            <div className="bg-white p-3 rounded border border-gray-300 text-sm text-gray-800 leading-snug">
              <strong className="text-purple-700">Key Advantages:</strong> Na-ion & Zn-ion use abundant materials (1000× more common than Li<sup className="text-blue-600">8</sup>), 
              eliminating supply chain risks. Solid-state doubles energy density and eliminates thermal runaway. All three are safer and more sustainable than current Li-ion.
            </div>
          </div>

          {/* Consequences */}
          <div className="bg-gradient-to-br from-red-50 to-orange-50 p-4 rounded-lg border border-red-300 shadow-sm">
            <h2 className="text-xl font-bold text-gray-900 mb-2 flex items-center">
              <span className="w-1 h-5 bg-red-600 mr-3"></span>
              Consequences of Inaction<sup className="text-xs text-blue-600">13,14</sup>
            </h2>
            <div className="grid grid-cols-2 gap-2 text-[13px] text-gray-800">
              <div className="bg-white p-2 rounded">💀 <strong>2.5M deaths/year</strong> from air pollution</div>
              <div className="bg-white p-2 rounded">🌡️ <strong>1.5°C budget</strong> exhausted in 4 years</div>
              <div className="bg-white p-2 rounded">💰 <strong>$1.09T losses</strong> from heat exposure</div>
              <div className="bg-white p-2 rounded">⚠️ <strong>2B people</strong> near toxic infrastructure</div>
            </div>
          </div>
        </div>

        {/* Right Column - 4 cols */}
        <div className="col-span-4 space-y-3">
          
          {/* Moss Landing */}
          <div className="bg-gradient-to-br from-blue-50 to-white p-4 rounded-lg border border-blue-200 shadow-sm">
            <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center">
              <span className="w-1 h-6 bg-blue-600 mr-3"></span>
              Moss Landing Energy Storage<sup className="text-xs text-blue-600">11,15</sup>
            </h2>
            <p className="text-xs text-gray-600 mb-3 italic">Monterey County, California • World's Largest Battery Installation</p>
            
            {/* Timeline Visualization */}
            <div className="relative mb-4">
              <div className="flex justify-between items-end mb-2">
                <div className="flex-1 text-center">
                  <div className="bg-blue-500 text-white px-3 py-2 rounded-t-lg font-bold text-sm">DEC 2020</div>
                  <div className="bg-blue-100 p-3 rounded-b-lg border-2 border-blue-500">
                    <div className="text-2xl font-black text-blue-900">300 MW</div>
                    <div className="text-xs text-gray-700">1,200 MWh</div>
                    <div className="text-xs text-gray-600 mt-1">Phase I</div>
                  </div>
                </div>
                <div className="w-8 flex items-center justify-center text-2xl text-gray-400">→</div>
                <div className="flex-1 text-center">
                  <div className="bg-blue-600 text-white px-3 py-2 rounded-t-lg font-bold text-sm">2021</div>
                  <div className="bg-blue-200 p-3 rounded-b-lg border-2 border-blue-600">
                    <div className="text-2xl font-black text-blue-900">400 MW</div>
                    <div className="text-xs text-gray-700">1,600 MWh</div>
                    <div className="text-xs text-gray-600 mt-1">Phase II</div>
                  </div>
                </div>
                <div className="w-8 flex items-center justify-center text-2xl text-gray-400">→</div>
                <div className="flex-1 text-center">
                  <div className="bg-blue-700 text-white px-3 py-2 rounded-t-lg font-bold text-sm">AUG 2023</div>
                  <div className="bg-blue-300 p-3 rounded-b-lg border-2 border-blue-700">
                    <div className="text-2xl font-black text-blue-900">750 MW</div>
                    <div className="text-xs text-gray-700">3,000 MWh</div>
                    <div className="text-xs text-gray-600 mt-1">Phase III</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-2 text-xs">
              <div className="bg-green-100 p-2 rounded text-center border border-green-300">
                <div className="font-black text-green-900">🌍 World's</div>
                <div className="text-gray-700">Largest</div>
              </div>
              <div className="bg-yellow-100 p-2 rounded text-center border border-yellow-300">
                <div className="font-black text-yellow-900">$100M</div>
                <div className="text-gray-700">Annual Savings</div>
              </div>
              <div className="bg-purple-100 p-2 rounded text-center border border-purple-300">
                <div className="font-black text-purple-900">13 acres</div>
                <div className="text-gray-700">Total Land</div>
              </div>
              <div className="bg-red-100 p-2 rounded text-center border border-red-300">
                <div className="font-black text-red-900">150%</div>
                <div className="text-gray-700">3yr Growth</div>
              </div>
            </div>
          </div>

          {/* Land Use */}
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <h2 className="text-xl font-bold text-gray-900 mb-3 flex items-center">
              <span className="w-1 h-5 bg-green-600 mr-3"></span>
              Land Use Comparison (750 MW)
            </h2>
            
            <div className="space-y-3">
              <div className="bg-gradient-to-r from-purple-100 to-purple-50 p-4 rounded-lg border-2 border-purple-400">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <span className="text-3xl mr-3">☀️🔋</span>
                    <div>
                      <div className="text-lg font-black text-gray-900">Solar + Battery</div>
                      <div className="text-xs text-gray-600">Moss Landing Facility</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-black text-purple-900">13 acres</div>
                    <div className="text-xs text-gray-600 mt-1">🏈🏈🏈🏈🏈🏈🏈🏈🏈🏈</div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-gray-200 to-gray-100 p-4 rounded-lg border-2 border-gray-400">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <span className="text-3xl mr-3">⛽🏭</span>
                    <div>
                      <div className="text-lg font-black text-gray-900">Coal Plant</div>
                      <div className="text-xs text-gray-600">Equivalent capacity</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-black text-gray-900">100+ acres</div>
                    <div className="text-xs text-gray-600 mt-1">🏈×77 + mining sites</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-3 bg-green-50 p-2 rounded border border-green-300 text-center">
              <div className="text-xs text-green-800 font-semibold">500× more land efficient</div>
            </div>
          </div>

          {/* Conclusion */}
          <div className="bg-gradient-to-br from-blue-600 via-purple-600 to-purple-700 p-4 rounded-lg shadow-lg text-white">
            <h2 className="text-2xl font-bold mb-3">Conclusion</h2>
            <ul className="space-y-2 text-[13px] leading-snug">
              <li className="flex items-start">
                <span className="mr-2 font-bold">•</span>
                <span><strong>Proven technology:</strong> 90% cost reduction since 2010, 3-10 year ROI, 86-90% emissions reduction vs fossil fuels</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2 font-bold">•</span>
                <span><strong>Real-world success:</strong> Moss Landing (750 MW, 13 acres) demonstrates scalability and efficiency</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2 font-bold">•</span>
                <span><strong>Next-gen solutions:</strong> Na-ion, Zn-ion, solid-state batteries eliminate material scarcity and safety risks</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2 font-bold">•</span>
                <span><strong>Economic opportunity:</strong> 30M jobs by 2030, $100M+ annual savings per facility</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2 font-bold">•</span>
                <span><strong>Urgent action needed:</strong> Delays perpetuate 2.5M deaths/year, $1T losses, and climate catastrophe</span>
              </li>
            </ul>
            <div className="mt-3 pt-3 border-t border-white/30 text-center">
              <p className="text-base font-bold">The technology is ready — deployment must accelerate now.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="h-[63px] bg-gray-100 border-t-2 border-gray-300 px-8 py-2">
        <div className="text-xs text-gray-700 leading-tight">
          <p className="font-bold mb-1">REFERENCES</p>
          <p className="leading-relaxed">
            [1] IRENA (2025) Battery Energy Storage Systems • [2] IEA (2024) Batteries and Secure Energy Transitions • [3] Wood Mackenzie (2025) Battery Energy Storage Comes of Age • 
            [4] Energy Storage Association (2024) Economic Benefits • [5] Climate Power (2025) Clean Energy Jobs Report • [6] MIT Climate Portal (2024) Battery Manufacturing CO₂ • 
            [7] IVL Swedish Environmental Research (2024) Battery Carbon Footprint • [8] Nature Communications (2024) Li vs Zn Availability • [9] Flinders University (2024) Zinc-Ion Breakthrough • 
            [10] ScienceDaily (2025) Sodium Batteries • [11] Vistra Energy (2023) Moss Landing Phase III • [12] Tesla (2025) Gigafactory Berlin Report • 
            [13] Lancet Countdown (2025) Health & Climate Change • [14] Global Carbon Project (2025) Fossil Fuel Emissions • [15] CPUC (2025) California Battery Progress
          </p>
        </div>
      </div>
    </div>
  );
};

export default BatteryPoster;