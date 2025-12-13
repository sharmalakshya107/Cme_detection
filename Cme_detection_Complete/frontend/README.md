# Frontend - CME Detection & Space Weather Monitoring Dashboard

**Made for Team Digi Shakti - Smart India Hackathon (SIH)**

This is the React-based frontend application that provides an interactive, real-time dashboard for space weather monitoring and CME detection. Built with modern web technologies for optimal performance and user experience.

## 🎯 Overview

The frontend is a single-page application (SPA) that provides:
- Real-time space weather data visualization
- Interactive 3D animations for space weather parameters
- CME detection and prediction interfaces
- Geomagnetic storm monitoring
- Satellite field data prediction
- Historical data analysis and charts
- Responsive design for all devices

## 🛠️ Tech Stack

### Core Framework
- **React 18.3.1**: Modern UI library with hooks and concurrent features
- **TypeScript**: Type-safe JavaScript for better development experience
- **Vite 5.4.1**: Lightning-fast build tool and dev server

### UI Libraries
- **Shadcn UI**: Beautiful, accessible component library built on Radix UI
- **Tailwind CSS 3.4.11**: Utility-first CSS framework
- **Framer Motion 12.23.24**: Smooth animations and page transitions
- **Lucide React**: Modern icon library

### Data Visualization
- **Recharts 2.12.7**: Composable charting library
- **Chart.js 4.5.1**: Flexible charting with react-chartjs-2
- **Three.js 0.160.0**: 3D graphics and animations
- **@react-three/fiber**: React renderer for Three.js
- **@react-three/drei**: Useful helpers for react-three/fiber

### State Management & Data Fetching
- **@tanstack/react-query 5.56.2**: Powerful data synchronization
- **React Router DOM 6.26.2**: Client-side routing

### Form Handling
- **React Hook Form 7.53.0**: Performant forms with easy validation
- **Zod 3.23.8**: TypeScript-first schema validation

## 📋 Prerequisites

- **Node.js 18+** and npm
- Modern web browser (Chrome, Firefox, Edge, Safari)
- Backend server running on `http://localhost:8002`

## 🚀 Installation

### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

### Step 2: Install Dependencies

```bash
npm install
```

### Step 3: Configure API Endpoint

Edit `src/lib/api.ts` to set the backend API URL:

```typescript
const API_BASE_URL = 'http://localhost:8002';
```

## 🏃 Running the Application

### Development Mode

```bash
npm run dev
```

The application will start on `http://localhost:8080`

### Production Build

```bash
npm run build
```

Build output will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── App.tsx                 # Main app component with routing
│   ├── main.tsx               # Application entry point
│   ├── index.css              # Global styles and space theme
│   │
│   ├── pages/                 # Page components
│   │   ├── Index.tsx          # Main dashboard
│   │   ├── Phase1.tsx         # Live space weather data
│   │   ├── Phase2.tsx         # CME prediction
│   │   ├── Phase3.tsx         # Live geomagnetic storm
│   │   ├── Phase4.tsx         # Geomagnetic storm prediction
│   │   ├── Phase5.tsx         # Video & image animation
│   │   ├── FieldDataPrediction.tsx  # Satellite CME prediction
│   │   ├── AllRecentCMEEvents.tsx   # Recent CME events
│   │   ├── NotFound.tsx       # 404 page
│   │   └── Phase1Helpers.ts   # Phase 1 helper functions
│   │
│   ├── components/            # Reusable components
│   │   ├── ui/                # Shadcn UI components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── dialog.tsx
│   │   │   └── ... (40+ components)
│   │   │
│   │   ├── ParticleDataChart.tsx        # Particle data visualization
│   │   ├── CMEDetectionPanel.tsx         # CME detection interface
│   │   ├── ParametersInfo.tsx            # Parameter information cards
│   │   ├── ForecastPredictionsPanel.tsx  # Forecast visualization
│   │   ├── DataImportExport.tsx           # Data import/export
│   │   ├── RawDataViewer.tsx              # Raw data display
│   │   ├── ParameterSpecificAnimation.tsx  # 3D parameter animations
│   │   ├── GeomagneticField3D.tsx         # 3D geomagnetic field
│   │   ├── SolarWind3D.tsx                # 3D solar wind visualization
│   │   ├── SpaceBackground.tsx            # Animated space background
│   │   ├── IntroAnimation.tsx             # Landing page animation
│   │   └── ... (more components)
│   │
│   ├── lib/                   # Utilities and configurations
│   │   ├── api.ts             # API client and endpoints
│   │   └── utils.ts           # Utility functions
│   │
│   └── hooks/                 # Custom React hooks
│       ├── use-mobile.tsx     # Mobile detection hook
│       └── use-toast.ts       # Toast notification hook
│
├── public/                    # Static assets
├── index.html                 # HTML template
├── package.json               # Dependencies and scripts
├── vite.config.ts             # Vite configuration
├── tailwind.config.ts         # Tailwind CSS configuration
├── tsconfig.json              # TypeScript configuration
└── postcss.config.js          # PostCSS configuration
```

## 🎨 Features

### Main Dashboard (`/`)
- Real-time mission status
- Key metrics overview
- System health monitoring
- Quick navigation to all phases
- Recent activity feed

### Phase 1: Live Space Weather Data (`/phase1`)
- **4-Grid Layout**:
  - Grid 1: Parameter info card with current values
  - Grid 2: Real-time 24-hour trend graph
  - Grid 3: 3D visualization/animations
  - Grid 4: Effects & safety analysis
- **15+ Parameters**: Kp, DST, Speed, Density, Bz, Bt, Temperature, etc.
- **Auto-refresh**: Updates every 60 seconds
- **Dynamic Analysis**: Real-time safety recommendations

### Phase 2: CME Prediction (`/phase2`)
- CME arrival time prediction
- Direction forecasting
- Forecast visualizations
- Historical comparison

### Phase 3: Live Geomagnetic Storm (`/phase3`)
- Real-time geomagnetic monitoring
- Storm intensity tracking
- Current storm effects
- Alert notifications

### Phase 4: Geomagnetic Storm Prediction (`/phase4`)
- Time regression models
- Storm intensity prediction
- Future timeline visualization
- Risk assessment

### Phase 5: Video & Image Animation (`/phase5`)
- Combined CME + Storm animations
- Video generation
- Image export capabilities
- GIF creation

### Field Data Prediction (`/phase`)
- Satellite selection by NORAD ID
- Coordinate matching with NOAA wind data
- CME probability calculation
- Risk level assessment
- Score breakdown visualization

### Recent CME Events (`/recent-cme-events`)
- Complete list of recent CME events
- Detailed detection results
- Event timeline
- Analysis summaries

## 🎨 Design System

### Color Scheme
- **Background**: Deep space blue-black (`#0B0B15`)
- **Primary**: Electric Purple (`#BD00FF`)
- **Secondary**: Cyan (`#00F0FF`)
- **Accent**: Hot Pink (`#FF0099`)
- **Neon Colors**: Blue, Pink, Purple, Green

### Typography
- **Font Family**: System fonts (San Francisco, Segoe UI, Roboto)
- **Headings**: Bold, gradient text with neon effects
- **Body**: Regular weight, high contrast for readability

### Components
- **Glassmorphism**: Frosted glass effect on cards
- **Neon Effects**: Glowing text and borders
- **Animations**: Smooth transitions with Framer Motion
- **3D Visualizations**: Interactive Three.js scenes

## 🔌 API Integration

### API Client Configuration

Located in `src/lib/api.ts`:

```typescript
const API_BASE_URL = 'http://localhost:8002';

export const api = {
  getDataSummary: () => fetch(`${API_BASE_URL}/api/data/summary`),
  getRealtimeData: () => fetch(`${API_BASE_URL}/api/data/realtime`),
  // ... more endpoints
};
```

### Data Fetching

Uses React Query for efficient data fetching:

```typescript
const { data, isLoading, error } = useQuery({
  queryKey: ['realtime'],
  queryFn: api.getRealtimeData,
  refetchInterval: 60000 // Refresh every 60 seconds
});
```

## 🎭 Animations

### Page Transitions
- Smooth page transitions using Framer Motion
- AnimatePresence for exit animations
- Route-based animation keys

### 3D Visualizations
- **Geomagnetic Field**: Rotating Earth with field lines
- **Solar Wind**: Particle streams from Sun to Earth
- **Parameter Animations**: Dynamic visualizations based on values

### Component Animations
- Staggered entrance animations
- Hover effects on cards
- Loading skeletons
- Toast notifications

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 768px (stacked layout)
- **Tablet**: 768px - 1024px (2-column layout)
- **Desktop**: > 1024px (4-grid layout)

### Mobile Optimizations
- Touch-friendly buttons
- Swipe gestures
- Optimized image loading
- Reduced animations on low-end devices

## 🔧 Configuration

### Vite Configuration (`vite.config.ts`)
- Port: 8080
- Host: `::` (all interfaces)
- Path aliases: `@/` → `src/`
- History API fallback for routing

### Tailwind Configuration (`tailwind.config.ts`)
- Custom space theme colors
- Extended animations
- Custom keyframes
- Dark mode support

### TypeScript Configuration
- Strict mode enabled
- Path aliases configured
- React JSX support

## 🧪 Development

### Code Structure
- **Components**: Reusable UI components
- **Pages**: Route-level components
- **Hooks**: Custom React hooks
- **Utils**: Helper functions
- **Lib**: External integrations

### Best Practices
- TypeScript for type safety
- Component composition
- Custom hooks for logic reuse
- Error boundaries for error handling
- Loading states for async operations

## 🐛 Troubleshooting

### Port Already in Use
Vite will automatically use the next available port.

### Module Not Found
```bash
npm install
```

### Build Errors
- Check Node.js version (requires 18+)
- Clear `node_modules` and reinstall
- Check TypeScript errors

### API Connection Errors
- Ensure backend is running on port 8002
- Check CORS configuration in backend
- Verify API_BASE_URL in `src/lib/api.ts`

### 3D Performance Issues
- Reduce animation quality on low-end devices
- Disable shadows for better performance
- Use simpler geometries

## 🚀 Deployment

### Build for Production

```bash
npm run build
```

### Deploy to Static Hosting

The `dist/` folder can be deployed to:
- **Vercel**: Automatic deployment
- **Netlify**: Drag and drop `dist/` folder
- **GitHub Pages**: Use GitHub Actions
- **AWS S3**: Upload `dist/` contents

### Docker Deployment

```bash
docker build -t cme-frontend .
docker run -p 8080:80 cme-frontend
```

## 📊 Performance Optimization

- **Code Splitting**: Automatic route-based code splitting
- **Lazy Loading**: Images and components loaded on demand
- **Memoization**: React.memo for expensive components
- **Virtual Scrolling**: For large lists
- **Image Optimization**: WebP format support

## 🔒 Security

- **Input Validation**: All user inputs validated
- **XSS Protection**: React escapes content by default
- **HTTPS**: Use HTTPS in production
- **CORS**: Configured in backend

## 📚 Additional Resources

- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)
- [Three.js Documentation](https://threejs.org/)
- [Framer Motion Documentation](https://www.framer.com/motion/)

---

## 👥 Development Team

**Made for Team Digi Shakti - Smart India Hackathon (SIH)**

**End to end developed and produced by me with the help of teammate Akshat Sharma**

---

**Note**: This frontend is part of the CME Detection & Space Weather Monitoring System. For backend documentation, see `../backend/README.md`.

